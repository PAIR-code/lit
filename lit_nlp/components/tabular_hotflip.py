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
"""HotFlip generator for tabular datasets.

A HotFlip is defined as a counterfactual input that is acquired by manipulating
the original input features in order to obtain a different prediction.

In contrast to (1), this implementation does not require access to the model
gradients. Instead, it uses examples from a dataset in order to find a set of
closest counterfactuals. Next, the closest counterfactuals and the original
input are linearly interpolated in order to find even closer counterfactuals.
Only scalar features are used in the interpolation search.

Only scalar and categorical features are used for the search of counterfactuals.
The features of other types are always assigned the value of the original input.

The implementation supports both classification and regression models. In case
of a regression model, the caller can specify a threshold value that
represents a decision boundary between the 'true' and 'false' values. If the
threshold is not specified then value 0.0 is used as the threshold.

The caller of the generator can set the maximum number of features that can
differ from the original example in order for the counterfactual to qualify.
In addition, the caller can specify the desired number of counterfactuals to be
returned.

References:
(1) HotFlip: White-Box Adversarial Examples for Text Classification
    Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
    ACL 2018.
    https://www.aclweb.org/anthology/P18-2006/

"""

import collections
from typing import Any, cast, Dict, List, Optional, Text, Tuple

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import cf_utils
from lit_nlp.lib import utils
import numpy as np

JsonDict = lit_types.JsonDict

PREDICTION_KEY = 'Prediction key'
NUM_EXAMPLES_KEY = 'Number of examples'
NUM_EXAMPLES_DEFAULT = 5
MAX_FLIPS_KEY = 'Maximum number of columns to change'
MAX_FLIPS_DEFAULT = 3
REGRESSION_THRESH_KEY = 'Regression threshold'
REGRESSION_THRESH_DEFAULT = 0.0


class TabularHotFlip(lit_components.Generator):
  """The HotFlip generator for tabular data."""

  # Hold dataset statistics such as standard deviation for scalar features and
  # same value probabilities for categorical features.
  _datasets_stats: Dict[Text, Dict[Text, float]] = {}

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:

    # Perform validation and retrieve configuration.
    if not model:
      raise ValueError('Please provide a model for this generator.')

    config = config or {}
    num_examples = int(config.get(NUM_EXAMPLES_KEY, NUM_EXAMPLES_DEFAULT))
    max_flips = int(config.get(MAX_FLIPS_KEY, MAX_FLIPS_DEFAULT))

    pred_key = config.get(PREDICTION_KEY, '')
    regression_thresh = float(
        config.get(REGRESSION_THRESH_KEY, REGRESSION_THRESH_DEFAULT))

    dataset_name = config.get('dataset_name')
    if not dataset_name:
      raise ValueError('The dataset name must be in the config.')

    output_spec = model.output_spec()
    if not pred_key:
      raise ValueError('Please provide the prediction key.')
    if pred_key not in output_spec:
      raise ValueError('Invalid prediction key.')

    if (not (isinstance(output_spec[pred_key], lit_types.MulticlassPreds) or
             isinstance(output_spec[pred_key], lit_types.RegressionScore))):
      raise ValueError(
          'Only classification and regression models are supported')

    # Calculate dataset statistics if it has never been calculated. The
    # statistics include such information as 'standard deviation' for scalar
    # features and probabilities for categorical features.
    if dataset_name not in self._datasets_stats:
      self._calculate_stats(dataset, dataset_name)

    # Find predicted class of the original example.
    original_pred = list(model.predict([example]))[0]

    # Find dataset examples that are flips.
    filtered_examples = self._filter_ds_examples(
        dataset=dataset,
        dataset_name=dataset_name,
        model=model,
        reference_output=original_pred,
        pred_key=pred_key,
        regression_thresh=regression_thresh)

    # Calculate L1 distance between the example and all other entries in the
    # dataset.
    distances = []
    for ds_example in filtered_examples:
      distance, diff_fields = self._calculate_L1_distance(
          example, ds_example, dataset, dataset_name=dataset_name, model=model)
      if distance > 0:
        distances.append((distance, diff_fields, ds_example))

    # Order the dataset entries based on the distance to the given example.
    distances.sort(key=lambda e: e[0])

    # Iterate through the dataset records and look for candidate hot-flips.
    candidates: List[JsonDict] = []
    for distance, diff_fields, ds_example in distances:
      flips = self._find_hot_flips(
          target_example=example,
          ds_example=ds_example,
          features_to_consider=diff_fields,
          model=model,
          target_pred=original_pred,
          pred_key=pred_key,
          dataset=dataset,
          max_num_features=max_flips,
          regression_threshold=regression_thresh)

      if flips:
        candidates.extend(flips)
      else:
        diffs = self._get_number_of_feature_diffs(model, example, ds_example)
        if diffs <= max_flips:
          candidates.append(ds_example)

      if len(candidates) >= num_examples:
        break

    # Calculate distances for the found hot flips.
    distances = []
    for flip_example in candidates:
      distance, diff_fields = self._calculate_L1_distance(
          example_1=example,
          example_2=flip_example,
          dataset=dataset,
          dataset_name=dataset_name,
          model=model)
      if distance > 0:
        distances.append((distance, diff_fields, flip_example))

    # Order the dataset entries based on the distance to the given example.
    distances.sort(key=lambda e: e[0])

    if len(distances) > num_examples:
      distances = distances[0:num_examples]

    # e[2] contains the hot-flip examples in the distances list of tuples.
    return [e[2] for e in distances]

  def _filter_ds_examples(
      self,
      dataset: lit_dataset.Dataset,
      dataset_name: Text,
      model: lit_model.Model,
      reference_output: JsonDict,
      pred_key: Text,
      regression_thresh: Optional[float] = None) -> List[JsonDict]:
    """Reads all dataset examples and returns only those that are flips."""
    if not isinstance(dataset, lit_dataset.IndexedDataset):
      raise ValueError(
          'Only indexed datasets are currently supported by the TabularHotFlip'
          'generator.')

    dataset = cast(lit_dataset.IndexedDataset, dataset)
    indexed_examples = list(dataset.indexed_examples)
    filtered_examples = []
    preds = model.predict_with_metadata(
        indexed_examples, dataset_name=dataset_name)

    # Find all DS examples that are flips with respect to the reference example.
    for indexed_example, pred in zip(indexed_examples, preds):
      flip = cf_utils.is_prediction_flip(
          cf_output=pred,
          orig_output=reference_output,
          output_spec=model.output_spec(),
          pred_key=pred_key,
          regression_thresh=regression_thresh)
      if flip:
        candidate_example = indexed_example['data'].copy()
        self._find_dataset_parent_and_set(
            model_output_spec=model.output_spec(),
            pred_key=pred_key,
            dataset_spec=dataset.spec(),
            example=candidate_example,
            predicted_value=pred[pred_key])
        filtered_examples.append(candidate_example)
    return filtered_examples

  def config_spec(self) -> lit_types.Spec:
    return {
        NUM_EXAMPLES_KEY:
            lit_types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_FLIPS_KEY:
            lit_types.TextSegment(default=str(MAX_FLIPS_DEFAULT)),
        PREDICTION_KEY:
            lit_types.FieldMatcher(
                spec='output', types=['MulticlassPreds', 'RegressionScore']),
        REGRESSION_THRESH_KEY:
            lit_types.TextSegment(default=str(REGRESSION_THRESH_DEFAULT)),
    }

  def _find_hot_flips(
      self,
      target_example: JsonDict,
      ds_example: JsonDict,
      features_to_consider: List[Text],
      model: lit_model.Model,
      target_pred: JsonDict,
      pred_key: Text,
      dataset: lit_dataset.Dataset,
      max_num_features: int,
      regression_threshold: Optional[float] = None,
  ) -> List[JsonDict]:
    """Computes hot-flip examples for a given target example and DS example.

    Args:
      target_example: target example for which the counterfactuals should be
        found.
      ds_example: a dataset example that should be used as a starting point for
        the search.
      features_to_consider: the list of feature keys that can be changed during
        the search.
      model: model to use for getting predictions.
      target_pred: model prediction that corresponds to `target_example`.
      pred_key: the name of the field in model predictions that contains the
        prediction value for the counterfactual search.
      dataset: a dataset object that contains `ds_example`.
      max_num_features: the maximum number of features that can differ between
        the target example and a counterfactual.
      regression_threshold: the threshold to use if `model` is a regression
        model. This parameter is ignored for classification models.

    Returns:
      A list of hot-flip counterfactuals that satisfy the criteria.
    """
    candidates: List[JsonDict] = []
    # First try to find counterfactuals with minimum number of feature flips
    # and increase the number of features if unsuccessful.
    for num_features in range(1, max_num_features + 1):
      feature_combinations = utils.find_all_combinations(
          features_to_consider,
          min_element_count=num_features,
          max_element_count=num_features)
      for feature_combination in feature_combinations:
        # All features other than the ones that are flipped should be assigned
        # the value of the target example.
        candidate_example = ds_example.copy()
        for field_name in target_example:
          if (field_name not in feature_combination and
              field_name in model.input_spec()):
            candidate_example[field_name] = target_example[field_name]
        flip, predicted_value = self._is_flip(
            model=model,
            cf_example=candidate_example,
            orig_output=target_pred,
            pred_key=pred_key,
            regression_thresh=regression_threshold)
        # Find closest flip by moving scalar values closer to the target.
        if flip:
          closest_flip = self._find_closest_flip(target_example,
                                                 candidate_example, target_pred,
                                                 pred_key, model, dataset,
                                                 regression_threshold)
          # If we found a closer flip through interpolation then add it,
          # otherwise add the previously found flip.
          if closest_flip is not None:
            candidates.append(closest_flip)
          else:
            self._find_dataset_parent_and_set(
                model_output_spec=model.output_spec(),
                pred_key=pred_key,
                dataset_spec=dataset.spec(),
                example=candidate_example,
                predicted_value=predicted_value)
            candidates.append(candidate_example)
      if candidates:
        return candidates
    return candidates

  def _find_closest_flip(self,
                         target_example: JsonDict,
                         example: JsonDict,
                         target_pred: JsonDict,
                         pred_key: Text,
                         model: lit_model.Model,
                         dataset: lit_dataset.Dataset,
                         regression_threshold: Optional[float] = None,
                         max_attempts: int = 4) -> Optional[JsonDict]:
    min_alpha = 0.0
    max_alpha = 1.0
    closest_flip = None
    input_spec = model.input_spec()
    for _ in range(max_attempts):
      current_alpha = (min_alpha + max_alpha) / 2
      candidate = example.copy()
      for field in target_example:
        if (field in candidate and field in input_spec and
            isinstance(input_spec[field], lit_types.Scalar) and
            candidate[field] is not None):
          candidate[field] = example[field] * (
              1 - current_alpha) + target_example[field] * current_alpha
      flip, predicted_value = self._is_flip(
          model=model,
          cf_example=candidate,
          orig_output=target_pred,
          pred_key=pred_key,
          regression_thresh=regression_threshold)
      if flip:
        self._find_dataset_parent_and_set(
            model_output_spec=model.output_spec(),
            pred_key=pred_key,
            dataset_spec=dataset.spec(),
            example=candidate,
            predicted_value=predicted_value)
        closest_flip = candidate
        min_alpha = current_alpha
      else:
        max_alpha = current_alpha
    return closest_flip

  def _is_flip(self,
               model: lit_model.Model,
               cf_example: JsonDict,
               orig_output: JsonDict,
               pred_key: Text,
               regression_thresh: Optional[float] = None) -> Tuple[bool, Any]:

    cf_output = list(model.predict([cf_example]))[0]
    feature_predicted_value = cf_output[pred_key]
    return cf_utils.is_prediction_flip(
        cf_output=cf_output,
        orig_output=orig_output,
        output_spec=model.output_spec(),
        pred_key=pred_key,
        regression_thresh=regression_thresh), feature_predicted_value

  def _find_fields(self, ds_spec: lit_dataset.Spec,
                   model_input_spec: lit_model.Spec):
    return set(ds_spec.keys()).intersection(model_input_spec.keys())

  def _calculate_stats(self, dataset: lit_dataset.Dataset,
                       dataset_name: Text) -> None:
    # Iterate through all examples in the dataset and store column values
    # in individual lists to facilitate future computation.
    field_values = {}
    spec = dataset.spec()
    for example in dataset.examples:
      for field_name in example:
        field_spec = spec[field_name]
        if not self._is_supported(field_spec):
          continue
        if example[field_name] is None:
          continue
        if field_name not in field_values:
          field_values[field_name] = []
        field_values[field_name].append(example[field_name])
    # Compute the necessary statistics: standard deviation for scalar fields and
    # probability of having same value for categorical and categorical fields.
    field_stats = {}
    for field_name, values in field_values.items():
      field_spec = spec[field_name]
      if self._is_scalar(field_spec):
        field_stats[field_name] = self._calculate_std_dev(values)
      elif self._is_categorical(field_spec):
        field_stats[field_name] = self._calculate_categorical_prob(values)
      else:
        assert False, 'Should never be reached.'
    # Cache the stats for the given dataset.
    self._datasets_stats[dataset_name] = field_stats

  def _calculate_std_dev(self, values: List[float]) -> float:
    return np.std(values)

  def _calculate_categorical_prob(self, values: List[float]) -> float:
    """Returns probability of two values from the list having the same value."""
    counts = collections.Counter(values)
    prob = 0.0
    for bucket in counts:
      prob += (counts[bucket] / len(values))**2
    return prob

  def _calculate_L1_distance(
      self, example_1: JsonDict, example_2: JsonDict,
      dataset: lit_dataset.Dataset, dataset_name: Text,
      model: lit_model.Model) -> Tuple[float, List[Text]]:
    """Calculates L1 distance between two input examples.

    Only categorical and scalar example features are considered. For categorical
    features, the distance is calculated as the probability of the feature
    having the same for two random (with replacement) examples. For scalar
    features, the unit of distance is equal to the standard deviation of all
    feature values.

    Only features that are in the intersection of the model and dataset features
    are considered.

    If a feature value of either of the examples is None, such feature is
    ignored in distance calculation and the name of the feature is not included
    in the result feature list (see Returns description).

    Args:
      example_1: a first example to measure distance for.
      example_2: a second example to measure distance for.
      dataset: a dataset that contains the information about the feature types.
      dataset_name: name of the dataset.
      model: a model that contains the information about the input feature
        types.

    Returns:
      A tuple that contains the L1 distance and the list of features that were
      used in the distance calculation. The list of features will only contain
    """
    distance = 0
    diff_fields = []
    fields = self._find_fields(
        ds_spec=dataset.spec(), model_input_spec=model.input_spec())
    for field_name in fields:
      field_spec = dataset.spec()[field_name]
      field_stats = self._datasets_stats[dataset_name]
      if not self._is_supported(field_spec):
        continue
      assert field_name in field_stats, f'{field_name}, {field_stats.keys()}'
      if example_1[field_name] == example_2[field_name]:
        continue
      if (example_1[field_name] is None) or (example_2[field_name] is None):
        continue
      diff_fields.append(field_name)
      if self._is_scalar(field_spec):
        std_dev = field_stats[field_name]
        if std_dev != 0:
          distance += abs(example_1[field_name] -
                          example_2[field_name]) / std_dev
      else:
        same_prob = field_stats[field_name]
        distance += same_prob
    return distance, diff_fields

  def _is_categorical(self, field_spec: lit_types.LitType) -> bool:
    """Checks whether a field is of categorical type."""
    return (isinstance(field_spec, lit_types.Boolean) or
            isinstance(field_spec, lit_types.CategoryLabel))

  def _is_scalar(self, field_spec: lit_types.LitType) -> bool:
    """Checks whether a field is of scalar type."""
    return isinstance(field_spec, lit_types.Scalar)

  def _is_supported(self, field_spec: lit_types.LitType) -> bool:
    """Checks whether a field should be used in distance calculation."""
    return self._is_scalar(field_spec) or self._is_categorical(field_spec)

  def _get_number_of_feature_diffs(self, model: lit_model.Model,
                                   example_1: JsonDict, example_2: JsonDict):
    """Counts the number of features that have different values in examples."""
    count = 0
    for feature in model.input_spec():
      if example_1[feature] != example_2[feature]:
        count += 1
    return count

  def _find_dataset_parent(self, model_output_spec: JsonDict, pred_key: Text,
                           dataset_spec: JsonDict) -> Optional[Text]:
    """Finds a field in dataset that is a parent of the model prediction."""
    output_feature = model_output_spec[pred_key]
    try:
      parent = output_feature.parent
      if parent not in dataset_spec:
        return None
      return parent
    except AttributeError:
      return None

  def _find_dataset_parent_and_set(self, model_output_spec: JsonDict,
                                   pred_key: Text, dataset_spec: JsonDict,
                                   example: JsonDict,
                                   predicted_value: Any) -> None:
    """Finds example parent field and assigns prediction value to it."""
    parent = self._find_dataset_parent(model_output_spec, pred_key,
                                       dataset_spec)
    if parent is not None:
      if isinstance(model_output_spec[pred_key], lit_types.MulticlassPreds):
        argmax = np.argmax(predicted_value)
        pred_field = cast(lit_types.MulticlassPreds,
                          model_output_spec[pred_key])
        label = pred_field.vocab[argmax]
        example[parent] = label
      else:
        example[parent] = predicted_value
