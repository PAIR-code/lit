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
"""Tests for lit_nlp.components.pdp."""

from typing import List

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import pdp
from lit_nlp.lib import caching
from lit_nlp.lib import testing_utils


JsonDict = lit_types.JsonDict


class TestRegressionPdp(lit_model.Model):

  def input_spec(self):
    return {'num': lit_types.Scalar(),
            'cats': lit_types.CategoryLabel(vocab=['One', 'None'])}

  def output_spec(self):
    return {'score': lit_types.RegressionScore()}

  def predict_minibatch(self, inputs: List[JsonDict], **kw):
    return [{'score': i['num'] + (1 if i['cats'] == 'One' else 0)}
            for i in inputs]


class TestClassificationPdp(lit_model.Model):

  def input_spec(self):
    return {'num': lit_types.Scalar(),
            'cats': lit_types.CategoryLabel(vocab=['One', 'None'])}

  def output_spec(self):
    return {'probas': lit_types.MulticlassPreds(vocab=['0', '1'])}

  def predict_minibatch(self, inputs: List[JsonDict], **kw):
    def pred(i):
      val = (i['num'] / 100) + (.5 if i['cats'] == 'One' else 0)
      return {'probas': [1 - val, val]}
    return [pred(i) for i in inputs]


class PdpTest(absltest.TestCase):

  def setUp(self):
    super(PdpTest, self).setUp()
    self.pdp = pdp.PdpInterpreter()
    self.reg_model = TestRegressionPdp()
    self.class_model = TestClassificationPdp()
    examples = [
        {
            'num': 1,
            'cats': 'One',
        },
        {
            'num': 10,
            'cats': 'None',
        },
        {
            'num': 5,
            'cats': 'One',
        },
    ]
    indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                      for ex in examples]
    self.dataset = lit_dataset.IndexedDataset(
        spec=self.reg_model.input_spec(), id_fn=caching.input_hash,
        indexed_examples=indexed_inputs)

  def test_regression_num(self):
    config = {
        'feature': 'num',
    }
    result = self.pdp.run_with_metadata([self.dataset.indexed_examples[0]],
                                        self.reg_model, self.dataset,
                                        config=config)
    expected = {1.0: 2.0, 2.0: 3.0, 3.0: 4.0, 4.0: 5.0, 5.0: 6.0, 6.0: 7.0,
                7.0: 8.0, 8.0: 9.0, 9.0: 10.0, 10.0: 11.0}
    testing_utils.assert_deep_almost_equal(self, result['score'], expected)

  def test_provided_range(self):
    config = {
        'feature': 'num',
        'range': [0, 9]
    }
    result = self.pdp.run_with_metadata([self.dataset.indexed_examples[0]],
                                        self.reg_model, self.dataset,
                                        config=config)
    expected = {0.0: 1.0, 1.0: 2.0, 2.0: 3.0, 3.0: 4.0, 4.0: 5.0, 5.0: 6.0,
                6.0: 7.0, 7.0: 8.0, 8.0: 9.0, 9.0: 10.0}
    testing_utils.assert_deep_almost_equal(self, result['score'], expected)

  def test_regression_cat(self):
    config = {
        'feature': 'cats',
    }
    result = self.pdp.run_with_metadata([self.dataset.indexed_examples[0]],
                                        self.reg_model, self.dataset,
                                        config=config)
    expected = {'One': 2.0, 'None': 1.0}
    testing_utils.assert_deep_almost_equal(self, result['score'], expected)

  def test_class_num(self):
    config = {
        'feature': 'num',
    }
    result = self.pdp.run_with_metadata([self.dataset.indexed_examples[0]],
                                        self.class_model, self.dataset,
                                        config=config)

    expected = {1.0: [0.49, 0.51], 2.0: [0.48, 0.52], 3.0: [0.47, 0.53],
                4.0: [0.46, 0.54], 5.0: [0.45, 0.55], 6.0: [0.44, 0.56],
                7.0: [0.43, 0.57], 8.0: [0.42, 0.58], 9.0: [0.41, 0.59],
                10.0: [0.4, 0.6]}
    testing_utils.assert_deep_almost_equal(self, result['probas'], expected)

  def test_classification_cat(self):
    config = {
        'feature': 'cats',
    }
    result = self.pdp.run_with_metadata([self.dataset.indexed_examples[0]],
                                        self.class_model, self.dataset,
                                        config=config)
    expected = {'One': [0.49, 0.51], 'None': [0.99, 0.01]}
    testing_utils.assert_deep_almost_equal(self, result['probas'], expected)

  def test_multiple_inputs(self):
    config = {
        'feature': 'num',
    }
    result = self.pdp.run_with_metadata(self.dataset.indexed_examples[0:2],
                                        self.reg_model, self.dataset,
                                        config=config)
    expected = {1.0: 1.5, 2.0: 2.5, 3.0: 3.5, 4.0: 4.5, 5.0: 5.5, 6.0: 6.5,
                7.0: 7.5, 8.0: 8.5, 9.0: 9.5, 10.0: 10.5}
    testing_utils.assert_deep_almost_equal(self, result['score'], expected)

if __name__ == '__main__':
  absltest.main()
