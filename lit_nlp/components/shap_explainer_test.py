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
# Lint as: python3
"""Tests for lit_nlp.components.shap_interpreter."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset
from lit_nlp.api import model
from lit_nlp.api import types
from lit_nlp.components import shap_explainer
from lit_nlp.lib import testing_utils


class EmptyModel(model.Model):

  def input_spec(self) -> types.Spec:
    return {}

  def output_spec(self) -> types.Spec:
    return {}

  def predict_minibatch(self, inputs, **kw):
    return None


class SparseMultilabelModel(testing_utils.TestIdentityRegressionModel):

  def output_spec(self) -> types.Spec:
    return {'preds': types.SparseMultilabelPreds()}

  def predict_minibatch(self, inputs, **kw):
    self.predict(inputs, **kw)

  def predict(self, inputs, **kw):
    return [{'preds': [('label', 0.8675309)]} for i in inputs]


class TabularShapExplainerTest(parameterized.TestCase):

  def setUp(self):
    super(TabularShapExplainerTest, self).setUp()
    self.classifier_model = testing_utils.TestModelClassification()
    self.dataset = dataset.Dataset(
        spec=self.classifier_model.input_spec(),
        examples=[{'score': 0.8675309}] * 10)
    self.empty_model = EmptyModel()
    self.regression_model = testing_utils.TestIdentityRegressionModel()
    self.shap = shap_explainer.TabularShapExplainer()
    self.sparse_model = SparseMultilabelModel()

  def test_compatibility(self):
    # Empty input and output specs
    self.assertFalse(self.shap.is_compatible(self.empty_model))
    # Contains input types that are not Scalar or CategoryLabel
    self.assertFalse(self.shap.is_compatible(self.classifier_model))
    # Good input and output spec
    self.assertTrue(self.shap.is_compatible(self.regression_model))
    self.assertTrue(self.shap.is_compatible(self.sparse_model))

  def test_valid_run(self):
    regress_salience = self.shap.run(
        self.dataset.examples,
        self.regression_model,
        self.dataset,
        config={'Prediction key': 'score'})
    sparse_salience = self.shap.run(
        self.dataset.examples,
        self.sparse_model,
        self.dataset,
        config={'Prediction key': 'preds'})
    self.assertEqual(len(self.dataset), len(regress_salience))
    self.assertEqual(len(self.dataset), len(sparse_salience))

  def test_run_without_pred_key(self):
    for cfg in (None, {}, {'Prediction key': None}, {'Prediction key': ''}):
      for mdl in (self.regression_model, self.sparse_model):
        self.assertIsNone(self.shap.run([], mdl, {}, config=cfg))

  def test_run_with_sample_size(self):
    regress_salience = self.shap.run(
        self.dataset.examples,
        self.regression_model,
        self.dataset,
        config={
            'Prediction key': 'score',
            'Sample size': 3
        })
    sparse_salience = self.shap.run(
        self.dataset.examples,
        self.sparse_model,
        self.dataset,
        config={
            'Prediction key': 'preds',
            'Sample size': 3
        })
    for salience in (regress_salience, sparse_salience):
      self.assertNotEqual(len(self.dataset), len(salience))
      self.assertLen(salience, 3)


if __name__ == '__main__':
  absltest.main()
