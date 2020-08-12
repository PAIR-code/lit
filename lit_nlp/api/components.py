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
from typing import Dict, List, Optional, Text

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types

JsonDict = types.JsonDict


class Interpreter(metaclass=abc.ABCMeta):
  """Base class for LIT interpretation components."""

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
                        indexed_inputs: List[JsonDict],
                        model: lit_model.Model,
                        dataset: lit_dataset.Dataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None):
    """Run this component, with access to data indices and metadata."""
    inputs = [ex['data'] for ex in indexed_inputs]
    return self.run(inputs, model, dataset, model_outputs, config)


class ComponentGroup(Interpreter):
  """Convenience class to package a group of components together."""

  def __init__(self, subcomponents: Dict[Text, Interpreter]):
    self._subcomponents = subcomponents

  def run_with_metadata(
      self,
      indexed_inputs: List[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
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


class Generator(metaclass=abc.ABCMeta):
  """Base class for LIT generators."""

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
