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
"""Base classes for LIT backend components."""
import abc
import inspect
from typing import Dict, List, Optional, Sequence, Text

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput


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
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    """Run this component, given a model and input(s)."""
    raise NotImplementedError(
        'Subclass should implement this, or override run_with_metadata() directly.'
    )

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None):
    """Run this component, with access to data indices and metadata."""
    inputs = [ex['data'] for ex in indexed_inputs]
    return self.run(inputs, model, dataset, model_outputs, config)

  def is_compatible(self, model: lit_model.Model):
    """Return if interpreter is compatible with the given model."""
    del model
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


class ComponentGroup(Interpreter):
  """Convenience class to package a group of components together."""

  def __init__(self, subcomponents: Dict[Text, Interpreter]):
    self._subcomponents = subcomponents

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[List[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Dict[Text, JsonDict]:
    """Run this component, given a model and input(s)."""
    assert model_outputs is not None
    assert len(model_outputs) == len(indexed_inputs)
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
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None):
    """Run this component, with access to data indices and metadata."""
    #  IndexedInput[] -> Input[]
    inputs = [ex['data'] for ex in indexed_inputs]
    return self.generate_all(inputs, model, dataset, config)

  def generate_all(self,
                   inputs: List[JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[JsonDict] = None) -> List[List[JsonDict]]:
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
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Return a list of generated examples."""
    return


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
  def annotate(self, inputs: List[JsonDict],
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
