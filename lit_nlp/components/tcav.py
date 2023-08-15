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
"""Quantitative Testing with Concept Activation Vectors (TCAV)."""

import math
import random
from typing import Any, cast, Optional, Sequence

import attr
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

import numpy as np
import scipy.stats
import sklearn.linear_model
import sklearn.model_selection

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec

NUM_SPLITS = 15  # TODO(lit-dev): Make this configurable in the UI.
RELATIVE_TCAV_SPLITS = [3, 5, 7, 10, 15]  # split sizes to try for relative TCAV
MIN_SPLIT_SIZE = 3
MIN_SPLITS = 2


@attr.s(auto_attribs=True, kw_only=True)
class TCAVConfig(object):
  """Config options for TCAV component."""
  concept_set_ids: list[str] = []
  class_to_explain: str = ''
  grad_layer: str = ''
  # Percentage of the example set to use in the test set when training the LM.
  test_size: Optional[float] = 0.33
  random_state: Optional[int] = 42
  negative_set_ids: list[str] = []
  # Optional pre-computed CAV to use by interpreter.
  cav: Optional[Any] = None


class TCAV(lit_components.Interpreter):
  """Quantitative Testing with Concept Activation Vectors (TCAV).

  TCAV is an interpretability method which allows users to define a concept
  by selecting a set of representative examples and trains a classifier at
  different model layers to determine whether that concept has any influence
  over the predictions of the model (https://arxiv.org/pdf/1711.11279.pdf).
  If so, it can be measured as an aggregate score, and also individual points
  can be measured by how much of that concept they contain, at a given layer.

  The original implementation can be found at:
  https://github.com/tensorflow/tcav

  This component requires that the following fields in the model spec. Field
  names like `layer` are placeholders; you can call them whatever you like,
  and as with other LIT components multiple segments are supported.
    Output:
      - Embeddings (`emb_layer`) to return the input embeddings
          for a layer
      - Gradients (`grad_layer`) to return the gradients w.r.t.
          `emb_layer`
      - Gradients class (`grad_class`) to return the label that gradients
        were computed for. This is usually a CategoryLabel, but can be anything
        since it will just be fed back into the model.
      - MulticlassPreds (`probas`)

  **Note: TCAV calls the model multiple times. It is _strongly_ recommended that
  you use a caching model (i.e., wrap your model in a `CachingModelWrapper`) to
  optimize performance.**
  """

  def hyp_test(self, scores, random_scores):
    """Returns the p-value for a two-sided t-test on the TCAV score."""
    # The null hypothesis is 0.5, since a TCAV score of 0.5 would indicate
    # the concept does not affect the prediction positively or negatively.
    _, p_val = scipy.stats.ttest_ind(scores, random_scores)
    return p_val

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused by TCAV
    output_spec = model.output_spec()
    gradient_fields = utils.find_spec_keys(output_spec, types.Gradients)

    if not gradient_fields:
      return False

    for grad_field in gradient_fields:
      field_spec = cast(types.Gradients, output_spec.get(grad_field))
      preds = output_spec.get(field_spec.align)
      compat_preds = isinstance(preds, types.MulticlassPreds)
      grad_for = output_spec.get(field_spec.grad_for)
      compat_grad_for = isinstance(grad_for, types.Embeddings)
      # TODO(b/205996131, b/294613507): remove grad_target_field_key and just
      # use the target labels from the input, similar to salience methods.
      grad_target = output_spec.get(field_spec.grad_target_field_key)
      compat_grad_target = isinstance(grad_target, types.CategoryLabel)
      if compat_preds and compat_grad_for and compat_grad_target:
        return True

    return False

  def get_predictions(
      self,
      model: lit_model.Model,
      inputs: Sequence[JsonDict],
      precomputed_outputs: Optional[Sequence[JsonDict]] = None,
  ) -> list[JsonDict]:
    """Get predictions with gradients w.r.t. argmax class."""
    # Get outputs using model.predict().
    if precomputed_outputs is None:
      predictions = list(model.predict(inputs))
    else:
      predictions = precomputed_outputs

    output_spec = model.output_spec()

    # TCAV always operates on the predicted class, but LIT models typically
    # compute gradients w.r.t. a specified target label.
    # Make new examples here with this set.
    target_fields = utils.find_spec_keys(output_spec, types.MulticlassPreds)
    valid_target_fields = [
        t for t in target_fields if getattr(output_spec[t], 'parent')
    ]
    modified_inputs = []
    for ex, preds in zip(inputs, predictions, strict=True):
      overrides = {}
      for field in valid_target_fields:
        label_idx = np.argmax(preds[field])
        label = cast(types.MulticlassPreds, output_spec[field]).vocab[label_idx]
        parent_field = getattr(output_spec[field], 'parent')
        overrides[parent_field] = label
      modified_inputs.append(utils.make_modified_input(ex, overrides, 'TCAV'))

    # TODO(b/294613507): enable caching here for better performance?
    # Any examples that were unmodified will already be cached, so this is only
    # for cases where argmax pred != target label.
    predictions = list(model.predict(modified_inputs))
    return predictions

  def run(
      self,
      inputs: Sequence[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Runs the TCAV method given the params in the inputs and config.

    Args:
      inputs: all examples in the dataset.
      model: the model being explained.
      dataset: the dataset which the current examples belong to.
      model_outputs: optional model outputs from calling model.predict(inputs).
      config: a config which should specify: {
          'concept_set_ids': [list of ids to use in concept set]
          'class_to_explain': [gradient class to explain],
          'grad_layer': [the Gradient field key of the layer to explain],
          'random_state': [an optional seed to make outputs deterministic]
          'test_size': [Percentage of the example set to use in the LM test set]
          'negative_set_ids': [optional list of ids to use as negative set] }

    Returns:
      A JsonDict containing the TCAV scores, directional derivatives,
      statistical test p-values, and LM accuracies.

    Raises:
      ValueError: configured `grad_layer` is not actually of type `Gradients`
    """
    if not config:
      raise TypeError('config must be provided')

    tcav_config = TCAVConfig(**(config or {}))
    # TODO(b/171513556): get these from the Dataset object once indices are
    # available there.
    dataset_examples = inputs

    # Get this layer's output spec keys for gradients and embeddings.
    grad_layer = tcav_config.grad_layer
    output_spec = model.output_spec()
    field_spec = output_spec.get(grad_layer)

    if not isinstance(field_spec, types.Gradients):
      raise ValueError(f'Configured grad_layer, {grad_layer}, must be a '
                       'Gradients field')

    field_spec = cast(types.Gradients, field_spec)
    emb_layer = field_spec.grad_for
    # Get the class that the gradients were computed for.
    grad_class_key = field_spec.grad_target_field_key

    predictions = self.get_predictions(model, dataset_examples, model_outputs)

    # If CAV is provided in config, then only calculate CAV similarity for
    # provided datapoints.
    if tcav_config.cav is not None:
      return [{
          'cos_sim':
              self._get_cos_sim(
                  np.array(tcav_config.cav), predictions, emb_layer)
      }]

    ids_set = set(tcav_config.concept_set_ids)
    concept_set_preds = [
        preds
        for ex, preds in zip(dataset_examples, predictions)
        if ex['_id'] in ids_set
    ]

    if tcav_config.negative_set_ids:
      negative_ids_set = set(tcav_config.negative_set_ids)
      negative_set_preds = [
          preds
          for ex, preds in zip(dataset_examples, predictions)
          if ex['_id'] in negative_ids_set
      ]
      return self._run_relative_tcav(
          grad_layer,
          emb_layer,
          grad_class_key,
          concept_set_preds,
          negative_set_preds,
          predictions,
          tcav_config,
      )
    else:
      non_concept_set_preds = [
          preds
          for ex, preds in zip(dataset_examples, predictions)
          if ex['_id'] not in ids_set
      ]
      return self._run_default_tcav(
          grad_layer,
          emb_layer,
          grad_class_key,
          concept_set_preds,
          non_concept_set_preds,
          predictions,
          tcav_config,
      )

  def _subsample(self, examples, n):
    return random.sample(examples, n) if n < len(examples) else examples

  def _run_default_tcav(
      self,
      grad_layer,
      emb_layer,
      grad_class_key,
      concept_outputs,
      non_concept_outputs,
      dataset_outputs,
      config,
  ):
    concept_results = []
    # If there are more concept set examples than non-concept set examples, we
    # use random splits of the concept examples as the concept set and use the
    # remainder of the dataset as the comparison set. Otherwise, we use random
    # splits of the non-concept examples as the comparison set.
    n = min(len(concept_outputs), len(non_concept_outputs))

    # If there are equal numbers of concept and non-concept examples, we
    # decrease n by one so that we also sample a different set in each TCAV run.
    if len(concept_outputs) == len(non_concept_outputs):
      n -= 1
    for _ in range(NUM_SPLITS):
      concept_split_outputs = self._subsample(concept_outputs, n)
      comparison_split_outputs = self._subsample(non_concept_outputs, n)
      concept_results.append(
          self._run_tcav(concept_split_outputs, comparison_split_outputs,
                         dataset_outputs, config.class_to_explain, emb_layer,
                         grad_layer, grad_class_key, config.test_size,
                         config.random_state))

    random_results = []
    # Get tcav scores on random splits.
    for _ in range(NUM_SPLITS):
      concept_split_outputs = self._subsample(dataset_outputs, n)
      comparison_split_outputs = self._subsample(non_concept_outputs, n)
      random_results.append(
          self._run_tcav(concept_split_outputs, comparison_split_outputs,
                         dataset_outputs, config.class_to_explain, emb_layer,
                         grad_layer, grad_class_key, config.test_size,
                         config.random_state))

    cav_scores = [res['score'] for res in concept_results]
    random_scores = [res['score'] for res in random_results]
    p_val = self.hyp_test(cav_scores, random_scores)

    random_mean = np.mean(random_scores)

    # Get index of CAV result with the highest accuracy.
    accuracies = [res['accuracy'] for res in concept_results]
    index = np.argmax(accuracies)

    # Many CAVS are trained and checked for statistical testing to calculate
    # the p-value. The values of the first CAV are returned.
    results = {
        'result': concept_results[index],
        'p_val': p_val,
        'random_mean': random_mean
    }
    return [results]

  def _run_relative_tcav(
      self,
      grad_layer,
      emb_layer,
      grad_class_key,
      positive_outputs,
      negative_outputs,
      dataset_outputs,
      config,
  ):
    # Ideally, for relative TCAV, users would test concepts with at least ~100
    # examples each so we can perform ~15 runs on unique subsets.
    # In practice, users may not pass in this many examples, so to accommodate
    # this, we use a cross-validation approach, where we will try different
    # subset split sizes, and return one with a statistically significant
    # result.
    splits = RELATIVE_TCAV_SPLITS
    min_length = min(len(positive_outputs), len(negative_outputs))

    # We set the minimum number of examples to run TCAV at 3 examples, and
    # need at least 2 runs for statistical testing. If there are too few
    # examples for this, we will perform 1 run of size
    # min(concept set length, negative set length), and return the result
    # without statistical testing.
    if (len(positive_outputs) < MIN_SPLIT_SIZE * MIN_SPLITS or
        len(negative_outputs) < MIN_SPLIT_SIZE * MIN_SPLITS):
      splits = [min_length]

    results = []
    for split in splits:
      num_runs = math.floor(min_length / split)

      # Exit if there are not enough examples for a run of this split size.
      if num_runs < 1:
        break

      concept_results = []
      # The i-th run will use the i-th (non-overlapping) subset of this split
      # size of examples.
      for i in range(num_runs):
        positive_split_outputs = positive_outputs[i * split: (i+1) * split]
        negative_split_outputs = negative_outputs[i * split: (i+1) * split]
        concept_results.append(
            self._run_tcav(positive_split_outputs, negative_split_outputs,
                           dataset_outputs, config.class_to_explain, emb_layer,
                           grad_layer, grad_class_key, config.test_size,
                           config.random_state))

      random_results = []
      # Get tcav scores on random splits.
      for _ in range(num_runs):
        positive_split_outputs = self._subsample(dataset_outputs, split)
        negative_split_outputs = self._subsample(dataset_outputs, split)
        random_results.append(
            self._run_tcav(positive_split_outputs, negative_split_outputs,
                           dataset_outputs, config.class_to_explain, emb_layer,
                           grad_layer, grad_class_key, config.test_size,
                           config.random_state))

      cav_scores = [res['score'] for res in concept_results]
      random_scores = [res['score'] for res in random_results]
      p_val = None
      if num_runs > 1:
        p_val = self.hyp_test(cav_scores, random_scores)

      random_mean = np.mean(random_scores)

      # Get index of CAV result with the highest accuracy.
      accuracies = [res['accuracy'] for res in concept_results]
      index = np.argmax(accuracies)

      # Many CAVS are trained and checked for statistical testing to calculate
      # the p-value. The values of the CAV with the highest accuracy for this
      # split is appended to the results.
      results.append({
          'result': concept_results[index],
          'p_val': p_val,
          'random_mean': random_mean,
          'split_size': split,
          'num_runs': num_runs,
      })

    tested_results = [
        result for result in results if result['p_val'] is not None
    ]
    # If there weren't enough runs for any t-testing, just return non-t-tested
    # results.
    if not tested_results:
      return results

    significant_tested_results = [
        result for result in tested_results if result['p_val'] < 0.05
    ]

    # If there were statistically significant results, return results from those
    # runs; otherwise, just return the (non-significant) t-tested results.
    if significant_tested_results:
      return significant_tested_results
    else:
      return tested_results

  def _get_training_data(self, comparison_outputs, concept_outputs, emb_layer):
    """Formats activations from model outputs as training data for the LM."""
    x = np.concatenate([[o[emb_layer] for o in comparison_outputs],
                        [o[emb_layer] for o in concept_outputs]])
    y = np.concatenate(
        [np.zeros(len(comparison_outputs)),
         np.ones(len(concept_outputs))])
    return x, y

  def _get_cos_sim(self, cav, datapoints_output: list[JsonDict],
                   emb_layer: str):
    cos_sim, _ = self.compute_local_scores(cav, datapoints_output, emb_layer)
    return cos_sim

  def _run_tcav(self,
                concept_outputs: list[JsonDict],
                comparison_outputs: list[JsonDict],
                dataset_outputs: list[JsonDict],
                class_to_explain: Any,
                emb_layer: str,
                grad_layer: str,
                grad_class_key: str,
                test_size: float,
                random_state=None):
    """Returns directional derivatives, tcav score, and LM accuracy."""
    x, y = self._get_training_data(comparison_outputs, concept_outputs,
                                   emb_layer)
    # Get CAV vector and accuracy of the trained linear model.
    cav, accuracy = self.get_trained_cav(x, y, test_size, random_state)

    # Compute directional derivatives for class to explain.
    dir_derivs = self.get_dir_derivs(cav, dataset_outputs, grad_layer,
                                     grad_class_key, class_to_explain)

    # Calculate the TCAV score using the gradient class directional derivatives.
    tcav_score = self.compute_tcav_score(dir_derivs)

    # Compute cosine similarity and dot product between CAV and activations.
    cos_sim, dot_prods = self.compute_local_scores(cav, dataset_outputs,
                                                   emb_layer)

    return {
        'score': tcav_score,
        'cos_sim': cos_sim,
        'dot_prods': dot_prods,
        'accuracy': accuracy,
        'cav': cav
    }

  def get_trained_cav(self, x, y, test_size, random_state=None):
    """Trains linear model on activations, returns weights (CAV) and accuracy."""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state)

    # Train linear model on training set.
    # TODO(b/177005822): Include additional linear classifier options
    # (e.g. L-BFGS, logistic regression, etc.)
    lm = sklearn.linear_model.SGDClassifier(random_state=random_state)
    lm.fit(x_train, y_train)
    cav = lm.coef_  # the weights of the LM are the CAV.

    # Compute accuracy on test set.
    y_pred = lm.predict(x_test)
    correct_count = 0
    for pred_val, test_val in zip(y_pred, y_test):
      if pred_val == test_val:
        correct_count += 1
    accuracy = correct_count / len(y_test)

    return cav, accuracy

  def get_dir_derivs(self, cav, dataset_outputs, grad_layer, grad_class_key,
                     class_to_explain):
    """Returns directional derivatives for class_to_explain examples."""
    dir_derivs = []

    for o in dataset_outputs:
      if o[grad_class_key] == class_to_explain:
        grad = o[grad_layer]
        # Multiplies the dataset_outputs’ gradients with the model’s weights
        # to get the directional derivative.
        dir_deriv = np.dot(grad, cav.flatten())
        dir_derivs.append(dir_deriv)
    return dir_derivs

  def compute_tcav_score(self, dir_derivs):
    """Returns the tcav score given the class to explain directional derivs."""
    # Maps positive derivatives to 1 and non-positive derivatives to 0.
    positive_dirs = [int(dir > 0) for dir in dir_derivs]

    # Divides the number of examples in the class_to_explain with directional
    # derivative > 0 by the total number of examples in the class_to_explain
    # to compute TCAV score.
    num_positive_dirs = sum(positive_dirs)
    return num_positive_dirs / len(dir_derivs)

  def compute_local_scores(self, cav, dataset_outputs, emb_layer):
    """Compute cosine similarity and dot product between CAV and activations."""
    flattened_cav = cav.flatten()
    dot_prods = [np.dot(flattened_cav, o[emb_layer]) for o in dataset_outputs]

    cav_magnitude = np.linalg.norm(flattened_cav)
    emb_magnitudes = [np.linalg.norm(o[emb_layer]) for o in dataset_outputs]
    cos_sim = [
        dot_prod / (emb_magnitude * cav_magnitude)
        for dot_prod, emb_magnitude in zip(dot_prods, emb_magnitudes)
    ]
    return cos_sim, dot_prods
