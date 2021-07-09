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
"""Threshold setter for binary classifiers."""

import math
from typing import cast, List, Optional, Sequence

import attr
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import metrics
import numpy as np


JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


@attr.s(auto_attribs=True, kw_only=True)
class TresholderConfig(object):
  """Config options for Thresholder component."""
  # Ratio of cost of a false negative to a false positive.
  cost_ratio: Optional[float] = 1

  # TODO(jwexler): Calculate optimal thresholds for faceted subgroups of
  # examples given the provided fairness strategies.
  faceted_examples: Optional[List[List[str]]] = []
  strategies: Optional[List[str]] = []


class Thresholder(lit_components.Interpreter):
  """Determines optimal thresholds for classifiers given constraints."""

  def __init__(self):
    self.metrics_gen = metrics.BinaryConfusionMetrics()

  def threshold_to_margin(self, thresh):
    # Convert between margin and classification threshold when displaying
    # margin as a threshold, as is done for binary classifiers.
    # Threshold is between 0 and 1 and represents the minimum score of the
    # positive (non-null) class before a datapoint is classified as positive.
    # A margin of 0 is the same as a threshold of .5 - meaning we take the
    # argmax class. A negative margin is a threshold below .5. Margin ranges
    # from -5 to 5, and can be converted the threshold through the equation
    # margin = ln(threshold / (1 - threshold)).
    if thresh == 0:
      return -5
    if thresh == 1:
      return 5
    return math.log(thresh / (1 - thresh))

  def get_cost(self, metrics_output, config):
    return metrics_output['FP'] * config.cost_ratio + metrics_output['FN']

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[List[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Calculates optimal thresholds on the provided data.

    Args:
      indexed_inputs: all examples in the dataset, in the indexed input format.
      model: the model being explained.
      dataset: the dataset which the current examples belong to.
      model_outputs: optional model outputs from calling model.predict(inputs).
      config: a config which should specify TresholderConfig options.

    Returns:
      A JsonDict containing the calcuated thresholds
    """
    config = TresholderConfig(**config) if config else TresholderConfig()

    pred_keys = []
    for pred_key, field_spec in model.output_spec().items():
      if self.metrics_gen.is_compatible(field_spec) and cast(
          types.MulticlassPreds, field_spec).parent:
        pred_keys.append(pred_key)

    # Try all margins for thresholds from 0 to 1, by hundreths.
    margins_to_try = [
        self.threshold_to_margin(t) for t in np.linspace(0, 1, 101)]

    # Get binary classification metrics for all margins.
    results = []
    for margin in margins_to_try:
      metrics_config = {}
      for pred_key in pred_keys:
        metrics_config[pred_key] = margin

      results.append(self.metrics_gen.run_with_metadata(
          indexed_inputs, model, dataset, model_outputs, metrics_config))

    # Find the threshold with the lowest cost
    pred_keys = [result['pred_key'] for result in results[0]]
    ret = []
    for i, pred_key in enumerate(pred_keys):
      metrics_list = [result[i]['metrics'] for result in results]
      costs = [self.get_cost(metrics_for_threshold, config)
               for metrics_for_threshold in metrics_list]
      best_idx = np.argmin(costs)
      # Divide best index by 100 to get threshold value between 0 and 1.
      ret.append({'pred_key': pred_key, 'threshold': best_idx / 100})
    return ret

