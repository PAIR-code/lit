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
"""Tests for lit_nlp.components.minimal_targeted_counterfactuals."""

from typing import List
import unittest.mock as mock

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import minimal_targeted_counterfactuals
from lit_nlp.lib import caching
import numpy as np
import scipy.special as scipy_special

ANIMALS = ['unknown', 'elephant', 'ant', 'whale', 'seal']


class ClassificationTestDataset(lit_dataset.Dataset):
  """A test dataset for classification testing."""

  def spec(self) -> lit_types.Spec:
    return {
        'size': lit_types.CategoryLabel(vocab=['small', 'medium', 'large']),
        'weight': lit_types.Scalar(),
        'legs': lit_types.Boolean(),
        'description': lit_types.String(),
        'animal': lit_types.CategoryLabel(vocab=ANIMALS),
    }

  @property
  def examples(self) -> List[lit_types.JsonDict]:
    return [
        {
            'size': 'small',
            'weight': 0.01,
            'legs': True,
            'description': 'small but strong',
            'animal': 'ant'
        },
        {
            'size': 'large',
            'weight': 0.8,
            'legs': True,
            'description': 'has a trunk',
            'animal': 'elephant'
        },
        {
            'size': 'medium',
            'weight': 0.2,
            'legs': False,
            'description': 'makes strange sounds',
            'animal': 'seal'
        },
        {
            'size': 'large',
            'weight': 2.5,
            'legs': False,
            'description': 'excellent water displacement',
            'animal': 'whale'
        },
    ]


class ClassificationTestModel(lit_model.Model):
  """A test model for testing tabular hot-flips on classification tasks."""

  def __init__(self, dataset: lit_dataset.Dataset) -> None:
    super().__init__()
    self._dataset = dataset

  def max_minibatch_size(self, **unused) -> int:
    return 2

  def input_spec(self) -> lit_types.Spec:
    return {
        'size': lit_types.CategoryLabel(vocab=['small', 'medium', 'large']),
        'weight': lit_types.Scalar(),
        'legs': lit_types.Boolean(),
        'description': lit_types.String(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        'preds':
            lit_types.MulticlassPreds(
                parent='animal', vocab=ANIMALS, null_idx=0)
    }

  def predict_minibatch(self, inputs: List[lit_types.JsonDict],
                        **unused) -> List[lit_types.JsonDict]:
    output = []

    def predict_example(ex: lit_types.JsonDict) -> lit_types.JsonDict:
      """Returns model predictions for a given example.

      The method uses the animal test dataset as the ground truth. The method
      compares the given example features to the dataset features for all
      animals. The closer the feature values are, the higher the contribution to
      the corresponding class logit is.

      Args:
        ex: an example to run prediction for.

      Returns:
        The softmax values for the animal class prediction.
      """
      # Logit values for ['unknown', 'elephant', 'ant', 'whale'].
      logits = np.zeros((len(ANIMALS),))
      for db_rec in self._dataset.examples:
        animal_index = ANIMALS.index(db_rec['animal'])
        for field_name in self._dataset.spec():
          if ex[field_name] is None or db_rec[field_name] is None:
            continue
          if field_name == 'animal':
            continue
          field_spec_value = self._dataset.spec()[field_name]
          if (isinstance(field_spec_value, lit_types.CategoryLabel) or
              isinstance(field_spec_value, lit_types.Boolean)) and (
                  ex[field_name] == db_rec[field_name]):
            logits[animal_index] += 1
          if isinstance(field_spec_value, lit_types.Scalar):
            logits[animal_index] += 1.0 - abs(ex[field_name] -
                                              db_rec[field_name])
      return scipy_special.softmax(logits)

    for example in inputs:
      output.append({'preds': predict_example(example)})
    return output


class RegressionTestDataset(lit_dataset.Dataset):
  """A test dataset for regression testing."""

  def spec(self) -> lit_types.Spec:
    return {
        'x_1': lit_types.Scalar(),
        'x_2': lit_types.Scalar(),
        'y': lit_types.Scalar(),
    }

  @property
  def examples(self) -> List[lit_types.JsonDict]:
    return [
        {
            'x_1': 0.0,
            'x_2': 0.0,
            'y': 0.0
        },
        {
            'x_1': 0.5,
            'x_2': 0.4,
            'y': 1.0
        },
    ]


class RegressionTestModel(lit_model.Model):
  """A test model for testing tabular hot-flips on regression tasks."""

  def max_minibatch_size(self, **unused) -> int:
    return 2

  def input_spec(self) -> lit_types.Spec:
    return {
        'x_1': lit_types.Scalar(),
        'x_2': lit_types.Scalar(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'score': lit_types.RegressionScore(parent='y')}

  def predict_minibatch(self, inputs: List[lit_types.JsonDict],
                        **unused) -> List[lit_types.JsonDict]:
    output = []

    def predict_example(ex: lit_types.JsonDict) -> lit_types.JsonDict:
      x1 = ex['x_1']
      x2 = ex['x_2']
      return 2 * x1**2 + x2

    for example in inputs:
      output.append({'score': predict_example(example)})
    return output


class ClassificationTabularMtcTest(absltest.TestCase):
  """Tests tabular hot-flips on classification tasks."""

  def setUp(self):
    super().setUp()
    dataset = lit_dataset.IndexedDataset(
        base=ClassificationTestDataset(), id_fn=caching.input_hash)
    self._dataset = dataset
    self._model = ClassificationTestModel(self._dataset)
    self._gen = minimal_targeted_counterfactuals.TabularMTC()

    self._example = {
        'size': 'large',
        'weight': 1.2,
        'legs': False,
        'description': 'big water animal',
        'animal': 'whale'
    }

    self._config = {
        'Prediction key': 'preds',
        'dataset_name': 'classification_test_dataset'
    }

  def test_test_model(self):
    """Tests the tests model predict method."""

    dataset = ClassificationTestDataset()
    model = ClassificationTestModel(dataset)
    preds = list(model.predict(dataset.examples))
    self.assertEqual(np.argmax(preds[0]['preds']), 2)
    self.assertEqual(np.argmax(preds[1]['preds']), 1)
    self.assertEqual(np.argmax(preds[2]['preds']), 4)
    self.assertEqual(np.argmax(preds[3]['preds']), 3)

  def test_prediction_key_required(self):
    """Tests the case when the client doesn't specify the prediction key."""
    self._config['Prediction key'] = ''
    with self.assertRaisesRegex(ValueError,
                                'Please provide the prediction key'):
      self._gen.generate(
          example=self._example,
          model=self._model,
          dataset=self._dataset,
          config=self._config)

  def test_incorrect_prediction_key(self):
    """Tests the case when the client specifies a key that doesn't exist."""
    self._config['Prediction key'] = 'wrong_key'
    with self.assertRaisesRegex(ValueError, 'Invalid prediction key'):
      self._gen.generate(
          example=self._example,
          model=self._model,
          dataset=self._dataset,
          config=self._config)

  def test_unsupported_model(self):
    """Tests the case when the passed model is not supported."""
    mocked_model = mock.MagicMock()
    output_spec = {'preds': lit_types.ImageBytes}
    mocked_model.output_spec = mock.MagicMock(return_value=output_spec)

    with self.assertRaisesRegex(
        ValueError, 'Only classification and regression models are supported'):
      self._gen.generate(
          example=self._example,
          model=mocked_model,
          dataset=self._dataset,
          config=self._config)

  def test_no_model(self):
    """Tests the case when no model is passed."""

    with self.assertRaisesRegex(ValueError,
                                'Please provide a model for this generator'):
      self._gen.generate(
          example=self._example,
          model=None,
          dataset=self._dataset,
          config=self._config)

  def test_max_number_of_records(self):
    """Tests that a client can specify a desired number of flips to return."""
    self._config['Number of examples'] = '2'
    result = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self.assertLen(result, 2)

  def test_text_fields_equal_to_target(self):
    """Tests that non-scalar non-categorical features has correct value.

    The values of non-scalar, non-categorical features should be the same as in
    the input example.
    """
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    s = {o['description'] for o in output}
    self.assertLen(s, 1)
    self.assertIn('big water animal', s)

  def test_mtc_prediction_is_argmax(self):
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    y_actual = output[0]['animal']
    y_expected = self._predict_and_return_argmax_label(output[0])
    self.assertEqual(y_actual, y_expected)

  def test_output_is_counterfactuals(self):
    """Tests that the returned values are indeed counterfactuals."""

    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self.assertGreaterEqual(len(output), 1)
    target_prediction = self._predict_and_return_argmax_label(self._example)
    for cf_example in output:
      cf_prediction = self._predict_and_return_argmax_label(cf_example)
      self.assertNotEqual(cf_prediction, target_prediction)

  def test_config_spec(self):
    """Tests that the generator returns spec with correct fields."""
    spec = self._gen.config_spec()
    self.assertIn('Number of examples', spec)
    self.assertIn('Maximum number of columns to change', spec)
    self.assertIn('Regression threshold', spec)
    self.assertIn('Prediction key', spec)

  def test_example_field_is_none(self):
    """Tests the case when a feature is assigned None value."""
    self._example['weight'] = None
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self.assertNotEmpty(output)

  def _predict_and_return_argmax_label(self, example):
    """Given an example, returns the index of the top prediction."""
    model_out = self._model.predict([example])
    softmax = list(model_out)[0]['preds']
    argmax = np.argmax(softmax)
    return self._model.output_spec()['preds'].vocab[argmax]


class RegressionTabularMtcTest(absltest.TestCase):
  """Tests tabular hot-flips with regression models."""

  def setUp(self):
    super().setUp()
    dataset = lit_dataset.IndexedDataset(
        base=RegressionTestDataset(), id_fn=caching.input_hash)
    self._dataset = dataset
    self._model = RegressionTestModel()
    self._gen = minimal_targeted_counterfactuals.TabularMTC()

    self._example = {'x_1': 1.0, 'x_2': 1.0}

    self._config = {
        'Prediction key': 'score',
        'dataset_name': 'regression_test_dataset'
    }

  def test_test_regression_model(self):
    """Tests the predict method of the regression model."""
    model = RegressionTestModel()
    example = {'x_1': 3, 'x_2': 2}
    pred = list(model.predict([example]))[0]
    self.assertEqual(pred['score'], 20)

  def test_output_is_below_threshold_counterfactuals(self):
    """Tests the case when the target prediction is above the threshold.

    If the target (reference) prediction is above the decision boundary
    threshold, the predictions for all counterfactuals should be below the
    threshold.
    """
    threshold = 2.8
    self._config['Regression threshold'] = str(threshold)
    self._example = {'x_1': 1.0, 'x_2': 1.0}
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    target_score = self._predict_and_return_score(self._example)
    self.assertGreaterEqual(target_score, threshold)
    self.assertNotEmpty(output)
    for cf_example in output:
      cf_score = self._predict_and_return_score(cf_example)
      self.assertLess(cf_score, threshold)

  def test_output_is_above_threshold_counterfactuals(self):
    """Tests the case when the target prediction is below the threshold.

    If the target (reference) prediction is below the decision boundary
    threshold, the predictions for all counterfactuals should be above or equal
    the threshold.
    """
    threshold = 0.1
    self._config['Regression threshold'] = str(threshold)
    self._example = {'x_1': 0.0, 'x_2': -5.0}
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    target_score = self._predict_and_return_score(self._example)
    self.assertLess(target_score, threshold)
    self.assertNotEmpty(output)
    for cf_example in output:
      cf_score = self._predict_and_return_score(cf_example)
      self.assertGreaterEqual(cf_score, threshold)

  def test_no_counterfactuals_found(self):
    """Tests the case when there no counterfactuals in the database."""
    threshold = 4.0
    self._config['Regression threshold'] = str(threshold)
    self._example = {'x_1': 1.0, 'x_2': 1.0}
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self.assertEmpty(output)

  def test_max_num_of_changed_columns(self):
    """Tests the client can set the number of features that can be changed."""
    self._config['Regression threshold'] = '0.25'
    self._config['Maximum number of columns to change'] = '1'
    self._example = {'x_1': 0.3, 'x_2': 0.3}
    output_1 = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self._config['Maximum number of columns to change'] = '2'
    output_2 = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    self.assertNotEmpty(output_1)
    self.assertNotEmpty(output_2)
    self.assertGreater(len(output_2), len(output_1))

  def test_parent_field_updated(self):
    threshold = 0.8
    self._config['Regression threshold'] = str(threshold)
    self._example = {'x_1': 0.0, 'x_2': 0.0}
    output = self._gen.generate(
        example=self._example,
        model=self._model,
        dataset=self._dataset,
        config=self._config)
    y_actual = output[0]['y']
    y_expected = self._predict_and_return_score(output[0])
    self.assertEqual(y_actual, y_expected)

  def _predict_and_return_score(self, example):
    """Given an example, returns the regression score."""
    model_out = self._model.predict([example])
    return list(model_out)[0]['score']


if __name__ == '__main__':
  absltest.main()
