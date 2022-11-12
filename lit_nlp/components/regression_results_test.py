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
"""Tests for lit_nlp.components.regression_results."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.components import regression_results
from lit_nlp.lib import testing_utils


class RegressionResultsTest(parameterized.TestCase):

  def setUp(self):
    super(RegressionResultsTest, self).setUp()
    self.interpreter = regression_results.RegressionInterpreter()

  @parameterized.named_parameters(
      ('classification', testing_utils.TestModelClassification(), False),
      ('regression', testing_utils.TestRegressionModel({}), True),
  )
  def test_is_compatible(self, model: lit_model.Model, epxected: bool):
    compat = self.interpreter.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, epxected)

  def test_run_with_label(self):
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {'label': 2}, {'label': -1}, {'label': 0}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestRegressionModel({}), dataset)
    expected = [
        {'scores': dtypes.RegressionResult(0, -2, 4)},
        {'scores': dtypes.RegressionResult(0, 1, 1)},
        {'scores': dtypes.RegressionResult(0, 0, 0)},
    ]
    self.assertEqual(expected, results)

  def test_run_with_no_label(self):
    dataset = lit_dataset.Dataset(None, None)
    inputs = [
        {}, {}, {}
    ]
    results = self.interpreter.run(
        inputs, testing_utils.TestRegressionModel({}), dataset)
    expected = [
        {'scores': dtypes.RegressionResult(0, None, None)},
        {'scores': dtypes.RegressionResult(0, None, None)},
        {'scores': dtypes.RegressionResult(0, None, None)},
    ]
    self.assertEqual(expected, results)

if __name__ == '__main__':
  absltest.main()
