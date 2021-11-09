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
  # Facets of datapoints to calculate individual thresholds for.
  facets: Optional[JsonDict] = {'': {}}


class Thresholder(lit_components.Interpreter):
  """Determines optimal thresholds for classifiers given constraints."""

  def __init__(self):
    self.metrics_gen = metrics.BinaryConfusionMetrics()

    # Set up the fairness measure calculators, which take confusion matrix
    # metrics and calculates a score from them.
    def demo_parity_metric(stats):
      return ((stats['TP'] + stats['FP']) /
              (stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']))

    def equal_acc_metric(stats):
      return ((stats['TP'] + stats['TN']) /
              (stats['TP'] + stats['FP'] + stats['TN'] + stats['FN']))

    def equal_opp_metric(stats):
      return stats['TP'] / (stats['TP'] + stats['FN'])

    self.fairness_measures = {
        'Demographic parity': demo_parity_metric,
        'Equal accuracy': equal_acc_metric,
        'Equal opporitunity': equal_opp_metric,
    }

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

  def get_cost(self, metrics_output, cost_ratio):
    return metrics_output['FP'] * cost_ratio + metrics_output['FN']

  def get_thresholds_for_pred_key(self, pred_key, i, margins_to_try,
                                  dataset_results, faceted_results, config):
    # Find the optimal threshold for the entire dataset
    metrics_list = [result[i]['metrics'] for result in dataset_results]
    costs = [self.get_cost(metrics_for_threshold, config.cost_ratio)
             for metrics_for_threshold in metrics_list]
    single_threshold = np.argmin(costs) / 100

    faceted_thresholds = {}
    faceted_costs = {}
    faceted_measures = {}

    # If there is only a single facet, return the single best threshold for
    # this prediction key.
    facets_keys = list(config.facets.keys())
    if len(facets_keys) == 1:
      faceted_thresholds[facets_keys[0]] = {
          'Single': single_threshold,
      }
      return {'pred_key': pred_key, 'thresholds': faceted_thresholds}

    # Loop through all facets of the dataset.
    for facet_key in facets_keys:
      # Find the optimal threshold for each facet individually.
      faceted_metrics_list = [
          result[i]['metrics'] for result in faceted_results[facet_key]]
      costs = [self.get_cost(metrics_for_threshold, config.cost_ratio)
               for metrics_for_threshold in faceted_metrics_list]
      ind_threshold = np.argmin(costs) / 100
      faceted_thresholds[facet_key] = {
          'Single': single_threshold,
          'Individual': ind_threshold
      }

      # Store the error costs and fairness measure scores for each measure for
      # the current facet across all possible margin values.
      measures = {}
      for measure_key in self.fairness_measures:
        measures[measure_key] = [
            self.fairness_measures[measure_key](metrics_for_threshold)
            for metrics_for_threshold in faceted_metrics_list]
      faceted_costs[facet_key] = costs
      faceted_measures[facet_key] = measures

    # Follows the same logic used in https://github.com/PAIR-code/what-if-tool
    # for calculating these thresholds.
    #
    # For all fairness measures:
    #   For all margins for first facet:
    #     For all other facets:
    #        Find margin with closest fairness measure of first facet at
    #        current margin
    #     Calculate overall cost for these margin settings settings across the
    #     facets.
    #   Save the margin settings that correspond to the lowest overall cost
    #   across faceted results.
    first_facet = facets_keys[0]
    for measure_key in self.fairness_measures:
      first_facet_costs = []
      first_facet_thresholds = []
      for threshold_idx in range(len(margins_to_try)):
        first_facet_measure = faceted_measures[
            first_facet][measure_key][threshold_idx]
        cost = faceted_costs[first_facet][threshold_idx]
        cur_thresholds = [threshold_idx / 100]
        for facet_to_check in facets_keys[1:]:
          distances_to_first_facet_measure = [
              abs(measure - first_facet_measure)
              for measure in faceted_measures[facet_to_check][measure_key]]
          threhold_idx_for_facet = np.argmin(distances_to_first_facet_measure)
          cost += faceted_costs[facet_to_check][threhold_idx_for_facet]
          cur_thresholds.append(threhold_idx_for_facet / 100)
        first_facet_costs.append(cost)
        first_facet_thresholds.append(cur_thresholds)
      min_measure_thresholds_idx = np.argmin(first_facet_costs)
      measure_thresholds = first_facet_thresholds[min_measure_thresholds_idx]
      for t_idx, facet_key in enumerate(facets_keys):
        faceted_thresholds[facet_key][measure_key] = measure_thresholds[t_idx]

    return {'pred_key': pred_key, 'thresholds': faceted_thresholds}

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

    indexed_outputs = {
        ex['id']: output for (ex, output) in zip(indexed_inputs, model_outputs)
    }

    # Try all margins for thresholds from 0 to 1, by hundreths.
    margins_to_try = [
        self.threshold_to_margin(t) for t in np.linspace(0, 1, 101)]

    # Get binary classification metrics for all margins, for the entire
    # dataset, and also for each facet specified in the config.
    dataset_results = []
    faceted_results = {}
    # Loop over each margin/threshold to check.
    for margin in margins_to_try:
      # Set up an empty config to pass to the metrics generator.
      metrics_config = {}
      for pred_key in pred_keys:
        metrics_config[pred_key] = {'': {'margin': margin}}

      # Get and store the metrics for the entire dataset for this margin.
      dataset_results.append(self.metrics_gen.run_with_metadata(
          indexed_inputs, model, dataset, model_outputs, metrics_config))

      # Get and store the metrics for each facet of the dataset for this margin.
      for facet_key in config.facets:
        if 'data' not in config.facets[facet_key]:
          continue
        if facet_key not in faceted_results:
          faceted_results[facet_key] = []
        faceted_model_outputs = [
            indexed_outputs[ex['id']]
            for ex in config.facets[facet_key]['data']]
        faceted_results[facet_key].append(self.metrics_gen.run_with_metadata(
            config.facets[facet_key]['data'], model, dataset,
            faceted_model_outputs, metrics_config))

    pred_keys = [result['pred_key'] for result in dataset_results[0]]
    ret = []

    # Find threshold information for each prediction key.
    for i, pred_key in enumerate(pred_keys):
      ret.append(self.get_thresholds_for_pred_key(
          pred_key, i, margins_to_try, dataset_results, faceted_results,
          config))
    return ret

