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
import inspect
from typing import Any, Optional, Sequence

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

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    """Run this component, given a model and input(s)."""
    raise NotImplementedError(
        'Subclass should implement this, or override run_with_metadata() directly.'
    )

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[list[JsonDict]] = None,
                        config: Optional[JsonDict] = None):
    """Run this component, with access to data indices and metadata."""
    inputs = [ex['data'] for ex in indexed_inputs]
    return self.run(inputs, model, dataset, model_outputs, config)

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


# TODO(b/254832560): Remove ComponentGroup class after promoting Metrics.
class ComponentGroup(Interpreter):
  """Convenience class to package a group of components together."""

  def __init__(self, subcomponents: dict[str, Interpreter]):
    self._subcomponents = subcomponents

  def meta_spec(self) -> types.Spec:
    spec: types.Spec = {}
    for component_name, component in self._subcomponents.items():
      for field_name, field_spec in component.meta_spec().items():
        spec[f'{component_name}: {field_name}'] = field_spec
    return spec

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> dict[str, JsonDict]:
    """Run this component, given a model and input(s)."""
    if model_outputs is None:
      raise ValueError('model_outputs cannot be None')

    if len(model_outputs) != len(indexed_inputs):
      raise ValueError('indexed_inputs and model_outputs must be the same size,'
                       f' received {len(indexed_inputs)} indexed_inputs and '
                       f'{len(model_outputs)} model_outputs')

    ret = {}
    for name, component in self._subcomponents.items():
      ret[name] = component.run_with_metadata(indexed_inputs, model, dataset,
                                              model_outputs, config)
    return ret


class Generator(Interpreter):
  """Base class for LIT generators."""

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[list[JsonDict]] = None,
                        config: Optional[JsonDict] = None):
    """Run this component, with access to data indices and metadata."""
    #  IndexedInput[] -> Input[]
    inputs = [ex['data'] for ex in indexed_inputs]
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
    """Return a list of generated examples."""
    pass


class Metrics(Interpreter):
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

  def run(
      self,
      inputs: Sequence[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> list[JsonDict]:
    raise NotImplementedError(
        'Subclass should implement its own run using compute.')

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> list[JsonDict]:
    inputs = [inp['data'] for inp in indexed_inputs]
    return self.run(inputs, model, dataset, model_outputs, config)

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
      config: Optional[JsonDict] = None) -> MetricsDict:
    """Compute metric(s) given labels and predictions."""
    raise NotImplementedError('Subclass should implement this, or override '
                              'compute_with_metadata() directly.')

  def compute_with_metadata(
      self,
      labels: Sequence[Any],
      preds: Sequence[Any],
      label_spec: types.LitType,
      pred_spec: types.LitType,
      indices: Sequence[types.ExampleId],
      metas: Sequence[JsonDict],
      config: Optional[JsonDict] = None) -> MetricsDict:
    """As compute(), but with access to indices and metadata."""
    del indices, metas  # unused by Metrics base class
    return self.compute(labels, preds, label_spec, pred_spec, config)


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
  def annotate(self, inputs: list[JsonDict],
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
