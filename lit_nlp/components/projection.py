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

from collections.abc import Hashable, Sequence
import threading
from typing import Optional

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import caching

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


class ProjectionInterpreter(lit_components.Interpreter):
  """Interpreter API implementation for dimensionality reduction model."""

  def __init__(
      self,
      model: lit_model.Model,
      inputs: Sequence[JsonDict],
      model_outputs: Optional[list[JsonDict]],
      projector: lit_model.ProjectorModel,
      field_name: str,
      name: str,
  ):
    self._projector = caching.CachingModelWrapper(projector, name=name)
    self._field_name = field_name

    # Train on the given examples
    self._run(model, inputs, model_outputs, do_fit=True)

  def is_compatible(
      self, model: lit_model.Model, dataset: lit_dataset.Dataset
  ) -> bool:
    del dataset, model  # Unused as field and model come through constructor
    return self._field_name in self._projector.output_spec()

  def convert_input(self, inp: JsonDict, model_output: JsonDict) -> JsonDict:
    """Convert inputs, preserving metadata for caching."""
    return {"x": model_output[self._field_name], "_id": inp.get("_id")}

  def _run(self,
           model: lit_model.Model,
           inputs: Sequence[JsonDict],
           model_outputs: Optional[Sequence[JsonDict]] = None,
           do_fit=False):
    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    converted_inputs = list(map(self.convert_input, inputs, model_outputs))
    if do_fit:
      return self._projector.fit_transform(converted_inputs)
    else:
      return self._projector.predict(converted_inputs)

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    """Run this component, given a model and input(s)."""
    del dataset   # Unused - Examples passed to constructor instead.
    # If using input values, then treat inputs as outputs instead of running
    # the model.
    if config and config.get("use_input"):
      model_outputs = inputs
    return self._run(model, inputs, model_outputs, do_fit=False)


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
  - (optional) proj_kw: config for the underlying ProjectorModel

  We also recommend including the model name and dataset name in the key, but
  this is not explicitly enforced.
  """

  def __init__(self, model_class: type[lit_model.ProjectorModel]):
    self._lock = threading.RLock()
    self._instances: dict[Hashable, ProjectionInterpreter] = {}
    # Used to construct new instances, given config['proj_kw']
    self._model_factory = model_class

  def _train_instance(
      self,
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      config: JsonDict,
      name: str
  ) -> ProjectionInterpreter:
    # Ignore pytype warning about abstract methods, since this should always
    # be a subclass of ProjectorModel which has these implemented.
    projector = self._model_factory(**config.get("proj_kw", {}))  # pytype: disable=not-instantiable
    train_inputs = dataset.examples

    # If using input values, then treat inputs as outputs instead of running
    # the model.
    if config.get("use_input"):
      train_outputs = train_inputs
    else:
      train_outputs = list(model.predict(train_inputs))
    logging.info("Creating new projection instance on %d points",
                 len(train_inputs))
    return ProjectionInterpreter(
        model,
        train_inputs,
        train_outputs,
        projector=projector,
        field_name=config["field_name"],
        name=name)

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    # UMAP code is not threadsafe and will throw strange 'index-out-of-bounds'
    # errors if multiple instances are accessed concurrently.
    with self._lock:
      instance_key = _key_from_dict(config)
      logging.info("Projection request: instance key: %s", instance_key)
      # Fit a new instance if necessary
      if instance_key not in self._instances:
        self._instances[instance_key] = self._train_instance(
            model, dataset, config, name=str(instance_key))

      proj_instance = self._instances[instance_key]
      # If projector was just trained, points should be cached.
      return proj_instance.run(inputs, model, dataset, model_outputs, config)
