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
"""Quantitative Testing with Concept Activation Vectors (TCAV)."""

import random
from typing import List, Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
import numpy as np
import scipy.stats
import sklearn.linear_model
import sklearn.model_selection


JsonDict = types.JsonDict
Spec = types.Spec

NUM_SPLITS = 20


class TCAV(lit_components.Interpreter):
  """Quantitative Testing with Concept Activation Vectors (TCAV).

  TCAV (https://arxiv.org/pdf/1711.11279.pdf) allows users to define a concept
  by selecting a set of representative examples and trains a classifier at
  different model layers to determine if that concept has any influence over
  the predictions of the model. If so, it can be measured as an aggregate score,
  and also individual points can be measured by how much of that concept they
  contain, at a given layer.

  The original implementation can be found at:
  https://github.com/tensorflow/tcav

  This component requires that the following fields in the model spec. Field
  names like `layer` are placeholders; you can call them whatever you like,
  and as with other LIT components multiple segments are supported.
    Output:
      - TokenEmbeddings (`layer`_cls_emb) to return the CLS input embeddings
          for a layer
      - TokenGradients (`layer`_cls_grad) to return the CLS gradients w.r.t.
          `layer`_cls_emb
      - MulticlassPreds ('probas')
  """

  def create_comparison_splits(self, dataset, concept_set,
                               num_splits=NUM_SPLITS):
    """Creates randomly sampled splits for multiple TCAV runs."""
    splits = []
    examples = list(dataset.examples)
    for _ in range(num_splits):
      splits.append(random.sample(examples, len(concept_set)))
    return splits

  def hyp_test(self, scores):
    """Returns the p value for a two-sided t-test on the TCAV score."""
    # The null hypothesis is 0.5, since a TCAV score of 0.5 would indicate
    # the concept does not affect the prediction positively or negatively.
    _, p_val = scipy.stats.ttest_1samp(scores, 0.5)
    return p_val

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Runs the TCAV method given the params in the inputs and config.

    Args:
      inputs: the examples to use for the 'concept' sets,
          following model.input_spec().
      model: the model being explained.
      dataset: the dataset which the current examples belong to.
      model_outputs: optional model outputs from calling model.predict(inputs).
      config: a config which should specify:
        {
          'concepts':
              {
                'concept1': ([start], [end]),
                'concept2': ([start], [end])
              },
          'gradient_class': [gradient class to explain],
          'layer': [layer to explain],
          'random_state': [an optional seed to make outputs deterministic]
        }

    Returns:
      A JsonDict containing the TCAV scores, directional derivatives,
      statistical test p-values, and LM accuracies.
    """
    random_state = None
    if 'random_state' in config:
      random_state = config['random_state']

    results = {}
    for concept in config['concepts']:
      # Parse concept examples from inputs using the config.
      start, end = config['concepts'][concept]
      concept_set = inputs[start:end]

      # Get outputs using model.predict().
      concept_outputs = list(model.predict(concept_set))
      dataset_outputs = list(model.predict(dataset.examples))

      # Create random splits of the dataset to use as comparison sets.
      splits = self.create_comparison_splits(dataset, concept_set)

      # Call run_tcav() on each comparison set.
      concept_results = []
      for comparison_set in splits:
        comparison_outputs = list(model.predict(comparison_set))
        concept_results.append(
            self._run_tcav(concept_set, concept_outputs, comparison_set,
                           comparison_outputs, dataset_outputs,
                           config['gradient_class'], config['layer'],
                           random_state))

      cav_scores = [res['score'] for res in concept_results]
      p_val = self.hyp_test(cav_scores)
      results[concept] = {'result': concept_results[0], 'p_val': p_val}
    return [results]

  def _get_training_data(self, comparison_outputs, concept_outputs, layer):
    """Formats activations from model outputs as training data for the LM."""
    x = []
    y = []
    for o in comparison_outputs:
      x.append(o['{}_cls_emb'.format(layer)])
      y.append(0)
    for o in concept_outputs:
      x.append(o['{}_cls_emb'.format(layer)])
      y.append(1)
    return x, y

  def _run_tcav(self, concept_set, concept_outputs, comparison_set,
                comparison_outputs, dataset_outputs, gradient_class,
                layer, random_state=None):
    """Returns directional derivatives, tcav score, and LM accuracy."""
    x, y = self._get_training_data(comparison_outputs, concept_outputs,
                                   layer)
    # Get CAV vector and accuracy of the trained linear model.
    cav, accuracy = self.get_trained_cav(x, y, random_state)

    # Compute directional derivatives for all dataset examples.
    dir_derivs, dir_derivs_gradient_class = self.get_dir_derivs(
        cav, dataset_outputs, layer, gradient_class)

    # Calculate the TCAV score using the gradient class directional derivatives.
    tcav_score = self.compute_tcav_score(dir_derivs_gradient_class)

    return {'score': tcav_score, 'dir_derivs': dir_derivs, 'accuracy': accuracy}

  def get_trained_cav(self, x, y, random_state=None):
    """Trains linear model on activations, returns weights (CAV) and accuracy."""
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33, stratify=y, random_state=random_state)

    # Train linear model on training set.
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

  def get_dir_derivs(self, cav, dataset_outputs, layer, gradient_class=1):
    """Returns directional derivatives for dataset and gradient_class examples."""
    dir_derivs = []
    dir_derivs_gradient_class = []

    for o in dataset_outputs:
      grad = o['{}_cls_grad'.format(layer)]
      # Multiplies the dataset_outputs’ gradients with the model’s weights
      # to get the directional derivative.
      dot = np.dot(grad, cav.flatten())
      dir_derivs.append(dot)

      if np.argmax(o['probas']) == gradient_class:
        dir_derivs_gradient_class.append(dot)
    return dir_derivs, dir_derivs_gradient_class

  def compute_tcav_score(self, dir_derivs_gradient_class):
    """Returns the tcav score given the gradient class directional derivatives."""
    # Maps positive derivatives to 1 and non-positive derivatives to 0.
    positive_dirs = [1 if dir > 0 else 0 for dir in dir_derivs_gradient_class]

    # Divides the number of examples in the class_to_explain with directional
    # derivative > 0 by the total number of examples in the class_to_explain
    # to compute TCAV score.
    num_positive_dirs = sum(positive_dirs)
    return num_positive_dirs / len(dir_derivs_gradient_class)
