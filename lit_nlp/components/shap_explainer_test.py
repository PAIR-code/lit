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

  def test_valid_run(self):
    regress_salience = self.shap.run(
        self.dataset.examples,
        self.regression_model,
        self.dataset,
        config={shap_explainer.EXPLAIN_KEY: 'score'})
    self.assertEqual(len(self.dataset), len(regress_salience))

  @parameterized.named_parameters(
      ('none', None),
      ('empty_dict', {}),
      ('empty_pred_key', {shap_explainer.EXPLAIN_KEY: ''}),
      ('none_pred_key', {shap_explainer.EXPLAIN_KEY: None}),
  )
  def test_config(self, config: dict[str, str]):
    self.assertIsNone(
        self.shap.run([], self.regression_model, _GOOD_DATASET, config=config))

  def test_run_with_sample_size(self):
    salience = self.shap.run(
        self.dataset.examples,
        self.regression_model,
        self.dataset,
        config={
            shap_explainer.EXPLAIN_KEY: 'score',
            shap_explainer.SAMPLE_KEY: 3
        })
    self.assertNotEqual(len(self.dataset), len(salience))
    self.assertLen(salience, 3)


if __name__ == '__main__':
  absltest.main()
