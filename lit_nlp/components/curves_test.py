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

from typing import NamedTuple
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import curves
from lit_nlp.lib import caching

# Labels used in the test dataset.
COLORS = ['red', 'green', 'blue']

_Curve = list[tuple[float, float]]
_Model = lit_model.BatchedModel


class _DataEntryForTesting(NamedTuple):
  prediction: tuple[float, float, float]
  label: str


TEST_DATA = {
    0: _DataEntryForTesting((0.7, 0.2, 0.1), 'red'),
    1: _DataEntryForTesting((0.3, 0.5, 0.2), 'red'),
    2: _DataEntryForTesting((0.6, 0.1, 0.3), 'blue'),
}


class _StaticTestModel(_Model):
  """A test model for the interpreter that uses 'TEST_DATA' as model output."""

  def input_spec(self) -> lit_types.Spec:
    return {'x': lit_types.Scalar()}

  def output_spec(self) -> lit_types.Spec:
    return {
        'pred': lit_types.MulticlassPreds(vocab=COLORS, parent='label'),
        'aux_pred': lit_types.MulticlassPreds(vocab=COLORS, parent='label')
    }

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict], **unused
  ) -> list[lit_types.JsonDict]:
    output = []

    def predict_example(ex: lit_types.JsonDict) -> tuple[float, float, float]:
      x = ex['x']
      return TEST_DATA[x].prediction

    for example in inputs:
      output.append({
          'pred': predict_example(example),
          'aux_pred': [1 / 3, 1 / 3, 1 / 3]
      })
    return output


class _IncompatiblePredictionTestModel(_Model):
  """A model with unsupported output type."""

  def input_spec(self) -> lit_types.Spec:
    return {'x': lit_types.Scalar()}

  def output_spec(self) -> lit_types.Spec:
    return {'pred': lit_types.RegressionScore(parent='label')}

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict], **unused
  ) -> list[lit_types.JsonDict]:
    return []


class _NoParentTestModel(_Model):
  """A model that doesn't specify the ground truth field in the dataset."""

  def input_spec(self) -> lit_types.Spec:
    return {'x': lit_types.Scalar()}

  def output_spec(self) -> lit_types.Spec:
    return {'pred': lit_types.MulticlassPreds(vocab=COLORS)}

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict], **unused_kw
  ) -> list[lit_types.JsonDict]:
    return []


class _StaticTestDataset(lit_dataset.Dataset):
  """Dataset for testing the interpreter that uses 'TEST_DATA' as the source."""

  def spec(self) -> lit_types.Spec:
    return {
        'x': lit_types.Scalar(),
        'label': lit_types.Scalar(),
    }

  @property
  def examples(self) -> list[lit_types.JsonDict]:
    return [{'x': x, 'label': entry.label} for x, entry in TEST_DATA.items()]


class CurvesInterpreterTest(parameterized.TestCase):
  """Tests CurvesInterpreter."""

  def setUp(self):
    super().setUp()
    self.dataset = lit_dataset.IndexedDataset(
        base=_StaticTestDataset(), id_fn=caching.input_hash
    )
    self.model = _StaticTestModel()
    self.ci = curves.CurvesInterpreter()

  def test_label_not_in_config(self):
    """The interpreter throws an error if the config doesn't have Label."""
    with self.assertRaises(ValueError):
      self.ci.run(
          inputs=self.dataset.examples,
          model=self.model,
          dataset=self.dataset,
      )

  def test_model_output_is_missing_in_config(self):
    """Tests the case when the name of the model output is absent in config.

    The interpreter throws an error if the name of the output is absent.
    """
    with self.assertRaises(ValueError):
      self.ci.run(
          inputs=self.dataset.examples,
          model=self.model,
          dataset=self.dataset,
          config={'Label': 'red'},
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='red',
          label='red',
          exp_roc=[(0.0, 0.0), (0.0, 0.5), (1.0, 0.5), (1.0, 1.0)],
          exp_pr=[(0.5, 0.5), (2 / 3, 1.0), (1.0, 0.5), (1.0, 0.0)],
      ),
      dict(
          testcase_name='blue',
          label='blue',
          exp_roc=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
          exp_pr=[(1.0, 1.0), (1.0, 0.0)],
      ),
  )
  def test_interpreter_honors_user_selected_label(
      self, label: str, exp_roc: _Curve, exp_pr: _Curve
  ):
    """Tests a happy scenario when a user doesn't specify the class label."""
    curves_data = self.ci.run(
        inputs=self.dataset.examples,
        model=self.model,
        dataset=self.dataset,
        config={
            curves.TARGET_LABEL_KEY: label,
            curves.TARGET_PREDICTION_KEY: 'pred',
        },
    )
    self.assertIn(curves.ROC_DATA, curves_data)
    self.assertIn(curves.PR_DATA, curves_data)
    self.assertEqual(curves_data[curves.ROC_DATA], exp_roc)
    self.assertEqual(curves_data[curves.PR_DATA], exp_pr)

  def test_config_spec(self):
    """Tests that the interpreter config has correct fields of correct type."""
    spec = self.ci.config_spec()
    self.assertIn(curves.TARGET_LABEL_KEY, spec)
    self.assertIsInstance(
        spec[curves.TARGET_LABEL_KEY], lit_types.CategoryLabel
    )
    self.assertIn(curves.TARGET_PREDICTION_KEY, spec)
    self.assertIsInstance(
        spec[curves.TARGET_PREDICTION_KEY], lit_types.SingleFieldMatcher
    )

  def test_meta_spec(self):
    """Tests that the interpreter meta has correct fields of correct type."""
    spec = self.ci.meta_spec()
    self.assertIn(curves.ROC_DATA, spec)
    self.assertIsInstance(spec[curves.ROC_DATA], lit_types.CurveDataPoints)
    self.assertIn(curves.PR_DATA, spec)
    self.assertIsInstance(spec[curves.PR_DATA], lit_types.CurveDataPoints)

  @parameterized.named_parameters(
      ('valid', _StaticTestModel(), True),
      ('no_multiclass_pred', _IncompatiblePredictionTestModel(), False),
      ('no_parent', _NoParentTestModel(), False),
  )
  def test_model_compatibility(self, model: _Model, exp_is_compat: bool):
    """A model is incompatible if prediction is not MulticlassPreds."""
    self.assertEqual(
        self.ci.is_compatible(model, _StaticTestDataset()), exp_is_compat
    )


if __name__ == '__main__':
  absltest.main()
