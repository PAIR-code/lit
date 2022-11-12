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
"""Tests for lit_nlp.components.curves."""

from typing import List, NamedTuple, Text, Tuple
from absl.testing import absltest
from absl.testing import parameterized

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.api.dataset import JsonDict
from lit_nlp.api.dataset import Spec
from lit_nlp.components import curves
from lit_nlp.lib import caching

# Labels used in the test dataset.
COLORS = ['red', 'green', 'blue']

Curve = list[tuple[float, float]]
Model = lit_model.Model


class TestDataEntry(NamedTuple):
  prediction: tuple[float, float, float]
  label: Text


TEST_DATA = {
    0: TestDataEntry((0.7, 0.2, 0.1), 'red'),
    1: TestDataEntry((0.3, 0.5, 0.2), 'red'),
    2: TestDataEntry((0.6, 0.1, 0.3), 'blue'),
}


class TestModel(Model):
  """A test model for the interpreter that uses 'TEST_DATA' as model output."""

  def input_spec(self) -> lit_types.Spec:
    return {
        'x': lit_types.Scalar(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        'pred': lit_types.MulticlassPreds(vocab=COLORS, parent='label'),
        'aux_pred': lit_types.MulticlassPreds(vocab=COLORS, parent='label')
    }

  def predict_minibatch(self, inputs: List[lit_types.JsonDict],
                        **unused) -> List[lit_types.JsonDict]:
    output = []

    def predict_example(ex: lit_types.JsonDict) -> Tuple[float, float, float]:
      x = ex['x']
      return TEST_DATA[x].prediction

    for example in inputs:
      output.append({
          'pred': predict_example(example),
          'aux_pred': [1 / 3, 1 / 3, 1 / 3]
      })
    return output


class IncompatiblePredictionTestModel(Model):
  """A model with unsupported output type."""

  def input_spec(self) -> lit_types.Spec:
    return {
        'x': lit_types.Scalar(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'pred': lit_types.RegressionScore(parent='label')}

  def predict_minibatch(self, inputs: List[lit_types.JsonDict],
                        **unused) -> List[lit_types.JsonDict]:
    return []


class NoParentTestModel(Model):
  """A model that doesn't specify the ground truth field in the dataset."""

  def input_spec(self) -> lit_types.Spec:
    return {
        'x': lit_types.Scalar(),
    }

  def output_spec(self) -> lit_types.Spec:
    return {'pred': lit_types.MulticlassPreds(vocab=COLORS)}

  def predict_minibatch(self, inputs: List[lit_types.JsonDict],
                        **unused) -> List[lit_types.JsonDict]:
    return []


class TestDataset(lit_dataset.Dataset):
  """Dataset for testing the interpreter that uses 'TEST_DATA' as the source."""

  def spec(self) -> Spec:
    return {
        'x': lit_types.Scalar(),
        'label': lit_types.Scalar(),
    }

  @property
  def examples(self) -> List[JsonDict]:
    data = []
    for x, entry in TEST_DATA.items():
      data.append({'x': x, 'label': entry.label})
    return data


class CurvesInterpreterTest(parameterized.TestCase):
  """Tests CurvesInterpreter."""

  def setUp(self):
    super().setUp()
    self.dataset = lit_dataset.IndexedDataset(
        base=TestDataset(), id_fn=caching.input_hash)
    self.model = TestModel()
    self.ci = curves.CurvesInterpreter()

  def test_label_not_in_config(self):
    """The interpreter throws an error if the config doesn't have Label."""
    with self.assertRaisesRegex(
        ValueError, 'The config \'Label\' field should contain the positive'
        ' class label.'):
      self.ci.run_with_metadata(
          indexed_inputs=self.dataset.indexed_examples,
          model=self.model,
          dataset=self.dataset,
      )

  def test_model_output_is_missing_in_config(self):
    """Tests the case when the name of the model output is absent in config.

    The interpreter throws an error if the name of the output is absent.
    """
    with self.assertRaisesRegex(
        ValueError, 'The config \'Prediction field\' should contain'):
      self.ci.run_with_metadata(
          indexed_inputs=self.dataset.indexed_examples,
          model=self.model,
          dataset=self.dataset,
          config={'Label': 'red'})

  @parameterized.named_parameters(
      ('red', 'red', [(0.0, 0.0), (0.0, 0.5), (1.0, 0.5), (1.0, 1.0)],
       [(0.5, 0.5), (2 / 3, 1.0), (1.0, 0.5), (1.0, 0.0)]),
      ('blue', 'blue', [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
       [(1.0, 1.0), (1.0, 0.0)]))
  def test_interpreter_honors_user_selected_label(
      self, label: str, exp_roc: Curve, exp_pr: Curve):
    """Tests a happy scenario when a user doesn't specify the class label."""
    curves_data = self.ci.run_with_metadata(
        indexed_inputs=self.dataset.indexed_examples,
        model=self.model,
        dataset=self.dataset,
        config={'Label': label, 'Prediction field': 'pred'})
    self.assertIn('roc_data', curves_data)
    self.assertIn('pr_data', curves_data)
    self.assertEqual(curves_data['roc_data'], exp_roc)
    self.assertEqual(curves_data['pr_data'], exp_pr)

  def test_config_spec(self):
    """Tests that the interpreter config has correct fields of correct type."""
    spec = self.ci.config_spec()
    self.assertIn('Label', spec)
    self.assertIsInstance(spec['Label'], lit_types.CategoryLabel)

  def test_meta_spec(self):
    """Tests that the interpreter meta has correct fields of correct type."""
    spec = self.ci.meta_spec()
    self.assertIn('roc_data', spec)
    self.assertIsInstance(spec['roc_data'], lit_types.CurveDataPoints)
    self.assertIn('pr_data', spec)
    self.assertIsInstance(spec['pr_data'], lit_types.CurveDataPoints)

  @parameterized.named_parameters(
      ('valid', TestModel(), True),
      ('no_multiclass_pred', IncompatiblePredictionTestModel(), False),
      ('no_parent', NoParentTestModel(), False))
  def test_model_compatibility(self, model: Model, exp_is_compat: bool):
    """A model is incompatible if prediction is not MulticlassPreds."""
    self.assertEqual(self.ci.is_compatible(model, TestDataset()), exp_is_compat)


if __name__ == '__main__':
  absltest.main()
