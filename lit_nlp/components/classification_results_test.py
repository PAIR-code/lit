# Copyright 2022 Google LLC
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
"""Tests for lit_nlp.components.classification_results."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.components import classification_results
from lit_nlp.lib import testing_utils
import numpy as np


class ClassificationResultsTest(parameterized.TestCase):

  def setUp(self):
    super(ClassificationResultsTest, self).setUp()
    self.interpreter = classification_results.ClassificationInterpreter()

  @parameterized.named_parameters(
      ('classification', testing_utils.TestModelClassification(), True),
      ('regression', testing_utils.TestRegressionModel({}), False),
  )
  def test_is_compatible(self, model: lit_model.Model, epxected: bool):
    compat = self.interpreter.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, epxected)

  def test_no_label(self):
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {}, {}, {}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestModelClassification(), dataset)
    expected = [
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', None)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', None)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', None)},
    ]
    self.assertListEqual(['probas'], list(results[0].keys()))
    for i in range(len(results)):
      np.testing.assert_array_equal(
          expected[i]['probas'].scores, results[i]['probas'].scores)
      self.assertEqual(
          expected[i]['probas'].predicted_class,
          results[i]['probas'].predicted_class)
      self.assertIsNone(results[i]['probas'].correct)

  def test_no_margins(self):
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {'label': '0'}, {'label': '1'}, {'label': '0'}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestModelClassification(), dataset)
    expected = [
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', False)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', True)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', False)},
    ]
    self.assertListEqual(['probas'], list(results[0].keys()))
    for i in range(len(results)):
      np.testing.assert_array_equal(
          expected[i]['probas'].scores, results[i]['probas'].scores)
      self.assertEqual(
          expected[i]['probas'].predicted_class,
          results[i]['probas'].predicted_class)
      self.assertEqual(
          expected[i]['probas'].correct,
          results[i]['probas'].correct)

  def test_single_margin(self):
    config = {'probas': {'': {'margin': 4, 'facetData': {'facets': {}}}}}
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {'label': '0'}, {'label': '1'}, {'label': '0'}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestModelClassification(), dataset, None, config)
    expected = [
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', True)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', False)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', True)},
    ]
    self.assertListEqual(['probas'], list(results[0].keys()))
    for i in range(len(results)):
      np.testing.assert_array_equal(
          expected[i]['probas'].scores, results[i]['probas'].scores)
      self.assertEqual(
          expected[i]['probas'].predicted_class,
          results[i]['probas'].predicted_class)
      self.assertEqual(
          expected[i]['probas'].correct,
          results[i]['probas'].correct)

  def test_faceted_margins_text(self):
    config = {'probas': {
        'hi': {'margin': 4, 'facetData': {'facets': {'s': {'val': 'hi'}}}},
        'bye': {'margin': -4, 'facetData': {'facets': {'s': {'val': 'bye'}}}}}}
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {'label': '0', 's': 'hi'}, {'label': '1', 's': 'hi'},
        {'label': '0', 's': 'bye'}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestModelClassification(), dataset, None, config)
    expected = [
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', True)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', False)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', False)},
    ]
    self.assertListEqual(['probas'], list(results[0].keys()))
    for i in range(len(results)):
      np.testing.assert_array_equal(
          expected[i]['probas'].scores, results[i]['probas'].scores)
      self.assertEqual(
          expected[i]['probas'].predicted_class,
          results[i]['probas'].predicted_class)
      self.assertEqual(
          expected[i]['probas'].correct,
          results[i]['probas'].correct)

  def test_faceted_margins_num(self):
    config = {'probas': {
        'high': {'margin': 4, 'facetData': {'facets': {'n': {'val': [2, 3]}}}},
        'low': {'margin': -4, 'facetData': {'facets': {'n': {'val': [0, 2]}}}}}}
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {'label': '0', 'n': 2.5}, {'label': '1', 'n': 2.1},
        {'label': '0', 'n': 1.5}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestModelClassification(), dataset, None, config)
    expected = [
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', True)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '0', False)},
        {'probas': dtypes.ClassificationResult([0.2, 0.8], '1', False)},
    ]
    self.assertListEqual(['probas'], list(results[0].keys()))
    for i in range(len(results)):
      np.testing.assert_array_equal(
          expected[i]['probas'].scores, results[i]['probas'].scores)
      self.assertEqual(
          expected[i]['probas'].predicted_class,
          results[i]['probas'].predicted_class)
      self.assertEqual(
          expected[i]['probas'].correct,
          results[i]['probas'].correct)

if __name__ == '__main__':
  absltest.main()
