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
"""Base classes for LIT backend components."""
import abc
from collections.abc import Sequence
import inspect
from typing import Any, Optional

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
MetricsDict = dict[str, float]


class Interpreter(metaclass=abc.ABCMeta):
  """Base class for LIT interpretation components."""

  def description(self) -> str:
    """Return a human-readable description of this component.

    Defaults to class docstring, but subclass may override this to be
    instance-dependent - for example, including the path from which the model
    was loaded.

    Returns:
      (string) A human-readable description for display in the UI.
    """
    return inspect.getdoc(self) or ''

  @abc.abstractmethod
  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    """Run this component, given a model and input(s)."""
    pass

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    """Return if interpreter is compatible with the dataset and model."""
    del dataset, model  # Unused in base class
    return True

  def config_spec(self) -> types.Spec:
    """Return the configuration spec for this component.

    If there are configuration options for this component that can be set in the
    UI, then list them and their type in this spec.

    Returns:
      Spec of configuration options. Defaults to an empty spec.
    """
    return {}

  def meta_spec(self) -> types.Spec:
    """Returns the metadata spec of this component.

    Can be used to represent information about what this interpreter returns,
    for use in the UI. For example, indicating if a saliency map is signed
    or unsigned which will affect the display of the results.

    Returns:
      A spec of what this component returns, to be used to drive the UI.
    """
    return {}


class Generator(Interpreter):
  """Base class for LIT generators."""

  def run(
      self,
      inputs: list[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None,
  ):
    del model_outputs
    return self.generate_all(inputs, model, dataset, config)

  def generate_all(self,
                   inputs: list[JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[JsonDict] = None) -> list[list[JsonDict]]:
    """Run generation on a set of inputs.

    Args:
      inputs: sequence of inputs, following model.input_spec()
      model: optional model to use to generate new examples.
      dataset: optional dataset which the current examples belong to.
      config: optional runtime config.

    Returns:
      list of list of new generated inputs, following model.input_spec()
    """
    output = []
    for ex in inputs:
      output.append(self.generate(ex, model, dataset, config))
    return output

  @abc.abstractmethod
  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> list[JsonDict]:
    """Return a list of generated examples, for a single input."""
    pass


class Metrics(Interpreter, metaclass=abc.ABCMeta):
  """Base class for LIT metrics components."""

  # Required methods implementations from Interpreter base class

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    """True if the model and dataset support metric computation."""
    for pred_spec in model.output_spec().values():
      parent_key: Optional[str] = getattr(pred_spec, 'parent', None)
      parent_spec: Optional[types.LitType] = dataset.spec().get(parent_key)
      if self.is_field_compatible(pred_spec, parent_spec):
        return True
    return False

  def meta_spec(self):
    """A dict of MetricResults defining the metrics computed by this class."""
    raise NotImplementedError('Subclass should define its own meta spec.')

  # New methods introduced by this subclass

  def is_field_compatible(
      self,
      pred_spec: types.LitType,
      parent_spec: Optional[types.LitType]) -> bool:
    """True if compatible with the prediction field and its parent."""
    del pred_spec, parent_spec  # Unused in base class
    raise NotImplementedError('Subclass should implement field compatibility.')

  def compute(
      self,
      labels: Sequence[Any],
      preds: Sequence[Any],
      label_spec: types.LitType,
      pred_spec: types.LitType,
      config: Optional[JsonDict] = None,
      indices: Optional[Sequence[types.ExampleId]] = None,
      metas: Optional[Sequence[JsonDict]] = None) -> MetricsDict:
    """Compute metric(s) given labels and predictions."""
    raise NotImplementedError('Subclass should implement this, or override '
                              'run() directly.')


class Annotator(metaclass=abc.ABCMeta):
  """Base class for LIT annotator components.

  Annotators are for adding extra fields to datapoints, using a model to
  annotate datapoints given their feature values.
  """

  def __init__(self, name: str, annotator_model: lit_model.Model):
    """Annotator constructor.

    Args:
      name: prinable name of the annotator, for use in new dataset fields.
      annotator_model: model to use to create dataset annotations.
    """
    self._name = name
    self._annotator_model = annotator_model

  @abc.abstractmethod
  def annotate(self, inputs: list[dict[str, Any]],
               dataset: lit_dataset.Dataset,
               dataset_spec_to_annotate: Optional[types.Spec] = None):
    """Annotate the provided inputs.

    Args:
      inputs: sequence of inputs, modified in-place.
      dataset: dataset which the examples belong to.
      dataset_spec_to_annotate: spec to add new annotated fields to, modified
        in-place. If none provided, then no spec is updated.

    Returns:
      Updated spec for the dataset, given the new annotations.
    """
    pass
