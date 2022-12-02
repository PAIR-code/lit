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
"""Partial dependence plot interpreter.

Runs a model on a set of edited examples to see the effect of changing a
specified feature on a set of examples. Returns a dict of prediction results for
different feature values, for each classification and regression head.

The front-end can display these as charts.
"""

import copy
import functools
from typing import cast, Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

_SUPPORTED_PRED_TYPES = (types.MulticlassPreds, types.RegressionScore)


class PdpInterpreter(lit_components.Interpreter):
  """Partial Dependence Plot interpreter."""

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused by PDP
    return utils.spec_contains(model.output_spec(), _SUPPORTED_PRED_TYPES)

  @functools.lru_cache()
  def get_vals_to_test(self, feat, dataset: lit_dataset.IndexedDataset):
    """Returns all values to try for a given feature."""
    numeric = isinstance(dataset.spec()[feat], types.Scalar)
    # Get categorical values to test from the spec vocab.
    if not numeric:
      return cast(types.MulticlassPreds, dataset.spec()[feat]).vocab or []

    # Create 10 values from min to max of a numeric feature to test and store
    # it for later use as well.
    nums = [ex[feat] for ex in dataset.examples]
    min_val = np.min(nums)
    max_val = np.max(nums)
    return np.linspace(min_val, max_val, 10)

  def run(self,
          inputs: list[types.JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[types.JsonDict]] = None,
          config: Optional[types.JsonDict] = None):
    """Create PDP chart info using provided inputs.

    Args:
      inputs: sequence of inputs, following model.input_spec()
      model: optional model to use to generate new examples.
      dataset: dataset which the current examples belong to.
      model_outputs: optional precomputed model outputs
      config: optional runtime config.

    Returns:
      A dict of alternate feature values to model outputs. The model outputs
      will be a number for regression models and a list of numbers for
      multiclass models.

    Raises:
      KeyError: `config` does not have a value for `feature`
      TypeError: `config` is missing
    """

    if not config:
      raise TypeError('config must be provided')

    feature = config.get('feature')
    if not feature:
      raise KeyError('Config must have a "feature" field')

    pred_keys = utils.find_spec_keys(model.output_spec(), _SUPPORTED_PRED_TYPES)
    if not pred_keys:
      logging.warning('PDP did not find any supported output fields.')
      return None

    provided_range = config.get('range', [])
    edited_outputs = {pred_key: {} for pred_key in pred_keys}

    # If a range was provided, use that to create the possible values.
    vals_to_test = (
        np.linspace(provided_range[0], provided_range[1], 10)
        if len(provided_range) == 2
        else self.get_vals_to_test(feature, dataset))

    # If no specific inputs provided, use the entire dataset.
    inputs_to_use = inputs if inputs else dataset.examples

    # For each alternate value for a given feature.
    for new_val in vals_to_test:
      # Create copies of all provided inputs with the value replaced.
      edited_inputs = []
      for inp in inputs_to_use:
        edited_input = copy.deepcopy(inp)
        edited_input[feature] = new_val
        edited_inputs.append(edited_input)

      # Run prediction on the altered inputs.
      outputs = list(model.predict(edited_inputs))

      # Store the mean of the prediction for the alternate value.
      for pred_key in pred_keys:
        field_spec = model.output_spec().get(pred_key)
        if isinstance(field_spec, types.RegressionScore):
          edited_outputs[pred_key][new_val] = np.mean(
              [output[pred_key] for output in outputs])
        else:
          edited_outputs[pred_key][new_val] = np.mean(
              [output[pred_key] for output in outputs], axis=0)

    return edited_outputs
