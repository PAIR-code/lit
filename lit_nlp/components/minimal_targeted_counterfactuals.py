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
"""Minimal Targeted Counterfactual generator for tabular datasets.

A Minimal Targeted Counterfactual is defined as a counterfactual input that is
acquired by manipulating the original input features in order to obtain a
different prediction.

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

The implementation aims to find the minimal set of counterfactuals. The set is
minimal if for each counterfactual in the set there exist no other
counterfactuals in the set that differ in the same features or subset of these
features and having smaller distance (cost) to the reference example. See (2)
for more details. The distances between two data points are measured as
described in (3).

The caller of the generator can set the maximum number of features that can
differ from the original example in order for the counterfactual to qualify.
In addition, the caller can specify the desired number of counterfactuals to be
returned.

References:
(1) HotFlip: White-Box Adversarial Examples for Text Classification.
    Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
    ACL 2018.
    https://www.aclweb.org/anthology/P18-2006/

(2) Local Explanations via Necessity and Sufficiency: Unifying Theory and
    Practice. David Watson, Limor Gultchin, Ankur Taly, Luciano Floridi.
    UAI 2021.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3825636
(3) The What-If Tool: Interactive Probing of Machine Learning Models.
    James Wexler, Mahima Pushkarna, Tolga Bolukbasi, Martin Wattenberg,
    Fernanda Viégas, Jimbo Wilson
    IEEE 2020.
    https://ieeexplore.ieee.org/abstract/document/8807255
"""

import collections
from typing import Any, cast, Dict, List, Optional, Text, Tuple
from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import cf_utils
from lit_nlp.lib import caching
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

# The maximum number of examples that will be searched per combination.
MAX_EXAMPLES_PER_COMBINATION = 50


class TabularMTC(lit_components.Generator):
  """The Minimal Targeted Counterfactual generator for tabular data.

  This generator looks for counterfactuals that are close to the original input
  without using gradients. First, the generator finds counterfactual examples
  from the dataset. Then, it changes features in the original input towards the
  counterfactual input until the decision boundary is found. During the search
  for a closer counterfactual, the algorithm manipulates different subsets of
  features, while keeping other features frozen. Thus, the algorithm tries to
  find the closest counterfactual that differs only in a single feature value,
  two feature values, etc., up to the selected "Maximum number of columns to
  change" configuration parameter. Only scalar and categorical features are
  changed during the search. The features of other types are always assigned the
  values of the original input.

  The implementation supports both classification and regression models. In case
  of a regression model, the "Regression threshold" parameter sets a decision
  boundary between the 'true' and 'false' values.

  The generator weakly guarantees that the result set of counterfactuals
  is minimal, i.e. for every counterfactual in the set, there exist no other
  counterfactual that differs in the same or smaller set of features and is
  closer to the original input.
  """

  # Hold dataset statistics such as standard deviation for scalar features and
  # the probability of having the same value as other random example for
  # categorical features. The outer dictionary key is the name of a dataset.
  # The nested dictionary key is the name of the example field. The value is
  # either the corresponding standard deviation for a scalar feature or the
  # probability for a categorical feature.
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

    supported_field_names = self._find_all_fields_to_consider(
        ds_spec=dataset.spec(),
        model_input_spec=model.input_spec(),
        example=example)

    candidates: List[JsonDict] = []

    # Iterate through all possible feature combinations.
    combs = utils.find_all_combinations(supported_field_names, 1, max_flips)
    for comb in combs:
      # Sort all dataset examples with respect to the given combination.
      sorted_examples = self._sort_and_filter_examples(
          examples=filtered_examples,
          ref_example=example,
          fields=comb,
          dataset=dataset,
          dataset_name=dataset_name)
      if not sorted_examples:
        continue

      # As an optimization trick, check whether the farthest example is a flip.
      # If it is not a flip then skip the current combination of features.
      # This optimization makes the minimum set guarantees weaker but
      # significantly improves the search speed.
      flip = self._find_hot_flip(
          ref_example=example,
          ds_example=sorted_examples[-1],
          features_to_consider=comb,
          model=model,
          target_pred=original_pred,
          pred_key=pred_key,
          dataset=dataset,
          interpolate=False,
          regression_threshold=regression_thresh)
      if not flip:
        logging.info('Skipped combination %s', comb)
        continue

      # Iterate through the sorted examples until the first flip is found.
      # TODO(b/204200758): improve performance by batching the predict requests.
      for ds_example in sorted_examples:
        flip = self._find_hot_flip(
            ref_example=example,
            ds_example=ds_example,
            features_to_consider=comb,
            model=model,
            target_pred=original_pred,
            pred_key=pred_key,
            dataset=dataset,
            interpolate=True,
            regression_threshold=regression_thresh)

        if flip:
          self._add_if_not_strictly_worse(
              example=flip,
              other_examples=candidates,
              ref_example=example,
              dataset=dataset,
              dataset_name=dataset_name,
              model=model)
          break

      if len(candidates) >= num_examples:
        break

    # Calculate distances for the found hot flips.
    candidate_tuples = []
    for flip_example in candidates:
      distance, diff_fields = self._calculate_L1_distance(
          example_1=example,
          example_2=flip_example,
          dataset=dataset,
          dataset_name=dataset_name,
          model=model)
      if distance > 0:
        candidate_tuples.append((distance, diff_fields, flip_example))

    # Order the dataset entries based on the distance to the given example.
    candidate_tuples.sort(key=lambda e: e[0])

    if len(candidate_tuples) > num_examples:
      candidate_tuples = candidate_tuples[0:num_examples]

    # e[2] contains the hot-flip examples in the distances list of tuples.
    return [e[2] for e in candidate_tuples]

  def _filter_ds_examples(
      self,
      dataset: lit_dataset.IndexedDataset,
      dataset_name: Text,
      model: lit_model.Model,
      reference_output: JsonDict,
      pred_key: Text,
      regression_thresh: Optional[float] = None) -> List[JsonDict]:
    """Reads all dataset examples and returns only those that are flips."""
    if not isinstance(dataset, lit_dataset.IndexedDataset):
      raise ValueError(
          'Only indexed datasets are currently supported by the TabularMTC'
          'generator.')

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
            lit_types.Scalar(
                min_val=1, max_val=20, default=NUM_EXAMPLES_DEFAULT, step=1),
        MAX_FLIPS_KEY:
            lit_types.Scalar(
                min_val=1, max_val=10, default=MAX_FLIPS_DEFAULT, step=1),
        PREDICTION_KEY:
            lit_types.FieldMatcher(
                spec='output', types=['MulticlassPreds', 'RegressionScore']),
        REGRESSION_THRESH_KEY:
            lit_types.TextSegment(default=str(REGRESSION_THRESH_DEFAULT)),
    }

  def _find_hot_flip(
      self,
      ref_example: JsonDict,
      ds_example: JsonDict,
      features_to_consider: List[Text],
      model: lit_model.Model,
      target_pred: JsonDict,
      pred_key: Text,
      dataset: lit_dataset.Dataset,
      interpolate: bool,
      regression_threshold: Optional[float] = None,
  ) -> Optional[JsonDict]:
    """Finds a hot-flip example for a given target example and DS example.

    Args:
      ref_example: target example for which the counterfactuals should be found.
      ds_example: a dataset example that should be used as a starting point for
        the search.
      features_to_consider: the list of feature keys that can be changed during
        the search.
      model: model to use for getting predictions.
      target_pred: model prediction that corresponds to `ref_example`.
      pred_key: the name of the field in model predictions that contains the
        prediction value for the counterfactual search.
      dataset: a dataset object that contains `ds_example`.
      interpolate: if True, the method tries to find a closer counterfactual
        using interpolation.
      regression_threshold: the threshold to use if `model` is a regression
        model. This parameter is ignored for classification models.

    Returns:
      A hot-flip counterfactual that satisfy the criteria.
    """
    # All features other than `features_to_consider` should be assigned the
    # value of the target example.
    candidate_example = ds_example.copy()
    for field_name in ref_example:
      if (field_name not in features_to_consider and
          field_name in model.input_spec()):
        candidate_example[field_name] = ref_example[field_name]

    flip, predicted_value = self._is_flip(
        model=model,
        cf_example=candidate_example,
        orig_output=target_pred,
        pred_key=pred_key,
        regression_thresh=regression_threshold)

    if not flip:
      return None

    # Find closest flip by moving scalar values closer to the target.
    closest_flip = None
    if interpolate:
      closest_flip = self._find_closer_flip_using_interpolation(
          ref_example, candidate_example, target_pred, pred_key, model, dataset,
          regression_threshold)
    # If we found a closer flip through interpolation then use it,
    # otherwise use the previously found flip.
    if closest_flip is not None:
      return closest_flip
    else:
      self._find_dataset_parent_and_set(
          model_output_spec=model.output_spec(),
          pred_key=pred_key,
          dataset_spec=dataset.spec(),
          example=candidate_example,
          predicted_value=predicted_value)
      return candidate_example

  def _find_closer_flip_using_interpolation(
      self,
      ref_example: JsonDict,
      known_flip: JsonDict,
      target_pred: JsonDict,
      pred_key: Text,
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      regression_threshold: Optional[float] = None,
      max_attempts: int = 4) -> Optional[JsonDict]:
    """Looks for the decision boundary between two examples using interpolation.

    The method searches for a flip that is closer to the `target example` than
    `known_flip`. The method performs the binary search by interpolating scalar
    values.

    Args:
      ref_example: an example for which the flip is searched.
      known_flip: an example that represents a known flip.
      target_pred: the model prediction at `ref_example`.
      pred_key: the named of the field inside `target_pred` that holds the
        prediction value.
      model: model to use for running predictions.
      dataset: dataset that contains `known_flip`.
      regression_threshold: threshold to use for regression models.
      max_attempts: number of binary search attempts.

    Returns:
      The counterfactual (flip) if found; 'None' otherwise.
    """
    min_alpha = 0.0
    max_alpha = 1.0
    closest_flip = None
    input_spec = model.input_spec()
    has_scalar = False
    for _ in range(max_attempts):
      # Interpolate the scalar values using binary search.
      current_alpha = (min_alpha + max_alpha) / 2
      candidate = known_flip.copy()
      for field in ref_example:
        if (field in candidate and field in input_spec and
            isinstance(input_spec[field], lit_types.Scalar) and
            candidate[field] is not None and ref_example[field] is not None):
          candidate[field] = known_flip[field] * (
              1 - current_alpha) + ref_example[field] * current_alpha
          has_scalar = True
      # The interpolation makes sense only for scalar values. If there are no
      # scalar fields that can be interpolated then terminate the search.
      if not has_scalar:
        return None
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

  def _find_all_fields_to_consider(
      self,
      ds_spec: lit_dataset.Spec,
      model_input_spec: lit_model.Spec,
      example: Optional[JsonDict] = None) -> List[Text]:
    overlapping = set(ds_spec.keys()).intersection(model_input_spec.keys())
    supported = [f for f in overlapping if self._is_supported(ds_spec[f])]
    if example:
      supported = [f for f in supported if example[f] is not None]
    return supported

  def _calculate_stats(self, dataset: lit_dataset.Dataset,
                       dataset_name: Text) -> None:
    # Iterate through all examples in the dataset and store column values
    # in individual lists to facilitate future computation.
    field_values = {}
    spec = dataset.spec()
    supported_fields = [name for name in spec if self._is_supported(spec[name])]
    for example in dataset.examples:
      for field_name in supported_fields:
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
      self,
      example_1: JsonDict,
      example_2: JsonDict,
      dataset: lit_dataset.Dataset,
      dataset_name: Text,
      model: Optional[lit_model.Model] = None,
      field_names: Optional[List[Text]] = None) -> Tuple[float, List[Text]]:
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
      field_names: if set then the distance calculation only considers these
        fields.

    Returns:
      A tuple that contains the L1 distance and the list of features that were
      used in the distance calculation. The list of features will only contain
    """
    assert model or field_names
    distance = 0
    diff_fields = []
    if field_names is None:
      assert model
      field_names = self._find_all_fields_to_consider(
          ds_spec=dataset.spec(), model_input_spec=model.input_spec())
    for field_name in field_names:
      field_spec = dataset.spec()[field_name]
      field_stats = self._datasets_stats[dataset_name]
      assert self._is_supported(field_spec)
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

  def _find_dataset_parent(self, model_output_spec: lit_types.Spec,
                           pred_key: Text,
                           dataset_spec: lit_types.Spec) -> Optional[Text]:
    """Finds a field in dataset that is a parent of the model prediction."""
    output_feature = model_output_spec[pred_key]
    parent = getattr(output_feature, 'parent', None)
    if parent not in dataset_spec:
      return None
    return parent

  def _find_dataset_parent_and_set(self, model_output_spec: lit_types.Spec,
                                   pred_key: Text, dataset_spec: lit_types.Spec,
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

  def _sort_and_filter_examples(self, examples: List[JsonDict],
                                ref_example: JsonDict, fields: List[Text],
                                dataset: lit_dataset.Dataset,
                                dataset_name: Text) -> List[JsonDict]:
    # Keep only those examples which field values are different from the
    # reference example.
    filtered_examples = []
    for example in examples:
      should_keep = True
      for field in fields:
        if example[field] == ref_example[field]:
          should_keep = False
          break
      if should_keep:
        filtered_examples.append(example)

    if not filtered_examples:
      return []

    # Deduplicate examples.
    dedup_hashes = set()
    dedup_examples = []
    for example in filtered_examples:
      h = self._create_hash(example, fields)
      if h not in dedup_hashes:
        dedup_examples.append(example)
        dedup_hashes.add(h)

    if len(dedup_examples) > MAX_EXAMPLES_PER_COMBINATION:
      dedup_examples = dedup_examples[:MAX_EXAMPLES_PER_COMBINATION]

    # Calculate distances with respect to the reference example taking into
    # consideration only the given fields.
    distances = []  # type: List[float]
    for example in dedup_examples:
      distance, _ = self._calculate_L1_distance(
          example_1=example,
          example_2=ref_example,
          dataset=dataset,
          dataset_name=dataset_name,
          field_names=fields)
      distances.append(distance)

    # Sort the filtered examples based on the distances.
    sorted_tuples = list(
        zip(*sorted(zip(dedup_examples, distances), key=lambda e: e[1])))[0]
    return list(sorted_tuples)

  def _add_if_not_strictly_worse(self, example: JsonDict,
                                 other_examples: List[JsonDict],
                                 ref_example: JsonDict,
                                 dataset: lit_dataset.Dataset,
                                 dataset_name: Text, model: lit_model.Model):
    for other_example in other_examples:
      is_worse = self._is_strictly_worse(
          example_1=example,
          example_2=other_example,
          ref_example=ref_example,
          dataset=dataset,
          dataset_name=dataset_name,
          model=model)
      if is_worse:
        return
    other_examples.append(example)

  def _is_strictly_worse(self, example_1: JsonDict, example_2: JsonDict,
                         ref_example: JsonDict, dataset: lit_dataset.Dataset,
                         dataset_name: Text, model: lit_model.Model) -> bool:
    """Calculates whether example_1 is strictly worse (or eq) than example_2."""

    # An example is strictly worse than other example if it differs in
    # the same or more features and has higher distance to the reference
    # example.
    ex_1_dist, ex_1_features = self._calculate_L1_distance(
        example_1=example_1,
        example_2=ref_example,
        dataset=dataset,
        dataset_name=dataset_name,
        model=model)
    ex_2_dist, ex_2_features = self._calculate_L1_distance(
        example_1=example_2,
        example_2=ref_example,
        dataset=dataset,
        dataset_name=dataset_name,
        model=model)
    if ex_1_dist < ex_2_dist:
      return False
    if set(ex_2_features).issubset(set(ex_1_features)):
      return True
    return False

  def _create_hash(self, example: JsonDict, fields: List[Text]) -> Text:
    json_map = {k: v for k, v in example.items() if k in fields}
    return caching.input_hash(json_map)
