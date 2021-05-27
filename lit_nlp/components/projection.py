# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Lint as: python3
"""Server-side dimensionality-reduction implementation.

This file implements machinery to manage dimensionality reduction models, such
as UMAP or PCA. Unlike most other LIT components, these projections need to be
learned on a particular dataset. This can take from a few seconds to a few
minutes, so for interactive use we want to cache both the projection and the
projected datapoints.

We implement this two-level caching with three classes:
- ProjectorModel simply wraps the projection model into a LIT Model, and
  provides training methods.
- ProjectionInterpreter implements the LIT Interpreter interface, and provides
  example-level caching.
- ProjectionManager implements the LIT Interpreter interface, and provides
  config-based caching of ProjectionInterpreter instances.

In most cases, the LIT server should interface with ProjectionManager so that
new configurations can be explored interactively (at the cost of re-training
projections).
"""

import abc
import copy
import threading
from typing import Any, Dict, List, Text, Optional, Hashable, Iterable, Type, Sequence

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import caching

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


class ProjectorModel(lit_model.Model, metaclass=abc.ABCMeta):
  """LIT model API implementation for dimensionality reduction."""

  ##
  # Training methods
  @abc.abstractmethod
  def fit_transform(self, inputs: Iterable[JsonDict]) -> List[JsonDict]:
    return

  def fit_transform_with_metadata(self, indexed_inputs) -> List[JsonDict]:
    return self.fit_transform((i["data"] for i in indexed_inputs))

  ##
  # LIT model API
  def input_spec(self):
    # 'x' denotes input features
    return {"x": types.Embeddings()}

  def output_spec(self):
    # 'z' denotes projected embeddings
    return {"z": types.Embeddings()}

  @abc.abstractmethod
  def predict_minibatch(self, inputs: Iterable[JsonDict],
                        **unused_kw) -> List[JsonDict]:
    return

  def max_minibatch_size(self, **unused_kw):
    return 1000


class ProjectionInterpreter(lit_components.Interpreter):
  """Interpreter API implementation for dimensionality reduction model."""

  def __init__(self, model: lit_model.Model,
               indexed_inputs: Sequence[IndexedInput],
               model_outputs: Optional[List[JsonDict]],
               projector: ProjectorModel, field_name: Text, name: Text):
    self._projector = caching.CachingModelWrapper(projector, name=name)
    self._field_name = field_name

    # Train on the given examples
    self._run(model, indexed_inputs, model_outputs, do_fit=True)

  def convert_input(self, indexed_input: JsonDict,
                    model_output: JsonDict) -> JsonDict:
    """Convert inputs, preserving metadata."""
    c = copy.copy(indexed_input)  # shallow copy
    c["data"] = {"x": model_output[self._field_name]}
    return c

  def _run(self,
           model: lit_model.Model,
           indexed_inputs: Sequence[IndexedInput],
           model_outputs: Optional[List[JsonDict]] = None,
           do_fit=False):
    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(indexed_inputs))
    assert len(model_outputs) == len(indexed_inputs)

    converted_inputs = list(
        map(self.convert_input, indexed_inputs, model_outputs))
    if do_fit:
      return self._projector.fit_transform_with_metadata(
          converted_inputs, dataset_name="")
    else:
      return self._projector.predict_with_metadata(
          converted_inputs, dataset_name="")

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.Dataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[Dict[Text, Any]] = None):
    del config  # unused - configure in constructor instead
    del dataset  # unused - pass examples to constructor instead
    return self._run(model, indexed_inputs, model_outputs, do_fit=False)


def _key_from_dict(d) -> Hashable:
  """Convert nested dict into a frozen, hashable structure usable as a key."""
  if isinstance(d, dict):
    return frozenset((k, _key_from_dict(v)) for k, v in d.items())
  elif isinstance(d, (list, tuple)):
    return tuple(map(_key_from_dict, d))
  else:
    return d


class ProjectionManager(lit_components.Interpreter):
  """Manager for multiple ProjectionInterpreter instances.

  Presents a standard "Interpreter" interface so that client code can treat
  this as an ordinary stateless component.

  The config is used to uniquely identify the projection instance, and a new
  instance is created and fit if not found.

  Config must contain the following fields:
  - field_name: name of embedding field (in model output)
  - (recommended) dataset_name: used for model cache
  - (optional) proj_kw: config for the underlying ProjectorModel

  We also recommend including the model name and dataset name in the key, but
  this is not explicitly enforced.
  """

  def __init__(self, model_class: Type[ProjectorModel]):
    self._lock = threading.RLock()
    self._instances = {}
    # Used to construct new instances, given config['proj_kw']
    self._model_factory = model_class

  def _train_instance(self, model: lit_model.Model,
                      dataset: lit_dataset.IndexedDataset, config: JsonDict,
                      name: Text) -> ProjectionInterpreter:
    # Ignore pytype warning about abstract methods, since this should always
    # be a subclass of ProjectorModel which has these implemented.
    projector = self._model_factory(**config.get("proj_kw", {}))  # pytype: disable=not-instantiable
    train_inputs = dataset.indexed_examples
    # TODO(lit-dev): remove 'dataset_name' from caching logic so we don't need
    # to track it here or elsewhere.
    train_outputs = list(
        model.predict_with_metadata(
            train_inputs, dataset_name=config.get("dataset_name")))
    logging.info("Creating new projection instance on %d points",
                 len(train_inputs))
    return ProjectionInterpreter(
        model,
        train_inputs,
        train_outputs,
        projector=projector,
        field_name=config["field_name"],
        name=name)

  def run_with_metadata(self, *args, **kw):
    # UMAP code is not threadsafe and will throw
    # strange 'index-out-of-bounds' errors if multiple instances are accessed
    # concurrently.
    with self._lock:
      return self._run_with_metadata(*args, **kw)

  def _run_with_metadata(self,
                         indexed_inputs: Sequence[IndexedInput],
                         model: lit_model.Model,
                         dataset: lit_dataset.IndexedDataset,
                         model_outputs: Optional[List[JsonDict]] = None,
                         config: Optional[Dict[Text, Any]] = None):
    instance_key = _key_from_dict(config)
    logging.info("Projection request: instance key: %s", instance_key)
    # Fit a new instance if necessary
    if instance_key not in self._instances:
      self._instances[instance_key] = self._train_instance(
          model, dataset, config, name=str(instance_key))

    proj_instance = self._instances[instance_key]
    # If projector was just trained, points should be cached.
    return proj_instance.run_with_metadata(indexed_inputs, model, dataset,
                                           model_outputs)
