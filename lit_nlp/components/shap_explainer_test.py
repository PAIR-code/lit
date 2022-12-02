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
"""Tests for lit_nlp.components.shap_interpreter."""

from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import shap_explainer
from lit_nlp.lib import testing_utils


_BAD_DATASET = lit_dataset.Dataset(
    spec={
        'segment': lit_types.TextSegment(),
        'scalar': lit_types.Scalar(required=False),
        'grad_class': lit_types.CategoryLabel(vocab=['0', '1'], required=False)
    },
    examples=[])

_GOOD_DATASET = lit_dataset.Dataset(
    spec={'val': lit_types.Scalar()}, examples=[{
        'val': 0.8675309
    }] * 10)


class EmptyModel(lit_model.Model):

  def input_spec(self) -> lit_types.Spec:
    return {}

  def output_spec(self) -> lit_types.Spec:
    return {}

  def predict_minibatch(self, inputs, **kw):
    return None


class SparseMultilabelModel(testing_utils.TestModelClassification):

  def output_spec(self) -> lit_types.Spec:
    return {'preds': lit_types.SparseMultilabelPreds()}

  def predict_minibatch(self, inputs, **kw):
    self.predict(inputs, **kw)

  def predict(self, inputs, **kw):
    return [{'preds': [('label', 0.8675309)]} for i in inputs]


class TabularShapExplainerTest(parameterized.TestCase):

  def setUp(self):
    super(TabularShapExplainerTest, self).setUp()
    self.dataset = _GOOD_DATASET
    self.regression_model = testing_utils.TestIdentityRegressionModel()
    self.shap = shap_explainer.TabularShapExplainer()
    self.sparse_model = SparseMultilabelModel()

  @parameterized.named_parameters(
      # Empty model always fails
      ('empty_model_bad_dataset', EmptyModel(), _BAD_DATASET, False),
      ('empty_model_good_dataset', EmptyModel(), _GOOD_DATASET, False),
      # Classification model never matches dataset
      ('cls_model_bad_dataset', testing_utils.TestModelClassification(),
       _BAD_DATASET, False),
      ('cls_model_good_dataset', testing_utils.TestModelClassification(),
       _GOOD_DATASET, False),
      # Incompatible dataset and model inut specs
      ('reg_model_bad_dataset', testing_utils.TestIdentityRegressionModel(),
       _BAD_DATASET, False),
      # Compatible dataset and model
      ('reg_model_good_dataset', testing_utils.TestIdentityRegressionModel(),
       _GOOD_DATASET, True),
      # Sparse model never matches dataset
      ('sparse_model_bad_dataset', SparseMultilabelModel(), _BAD_DATASET, False
      ),
      ('sparse_model_good_dataset', SparseMultilabelModel(), _GOOD_DATASET,
       False),
  )
  def test_compatibility(self, model: lit_model.Model,
                         dataset: lit_dataset.Dataset, expected: bool):
    self.assertEqual(self.shap.is_compatible(model, dataset), expected)

  def test_config_errors(self):
    with self.assertRaises(ValueError):
      self.shap.run([],
                    self.regression_model,
                    _GOOD_DATASET,
                    config={shap_explainer.EXPLAIN_KEY: 'not_in_mdl'})

  @parameterized.named_parameters(
      ('default', None, None),
      ('pred_key_only', 'score', None),
      ('sample_size_only', None, 3),
      ('all_fields', 'score', 3),
  )
  def test_run(self, pred_key: Optional[str], sample_size: Optional[int]):
    config = {}

    if pred_key:
      config[shap_explainer.EXPLAIN_KEY] = pred_key

    if sample_size:
      config[shap_explainer.SAMPLE_KEY] = sample_size
      expected_length = sample_size
    else:
      expected_length = len(self.dataset.examples)

    regress_salience = self.shap.run(
        self.dataset.examples,
        self.regression_model,
        self.dataset,
        config=config)
    self.assertLen(regress_salience, expected_length)


if __name__ == '__main__':
  absltest.main()
