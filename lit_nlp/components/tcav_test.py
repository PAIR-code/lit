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
"""Tests for lit_nlp.components.gradient_maps."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import tcav
import numpy as np


_TEST_VOCAB = ['0', '1']


class VariableOutputSpecModel(lit_model.BatchedModel):
  """A dummy model used for testing interpreter compatibility."""

  def __init__(self, output_spec: lit_types.Spec):
    self._output_spec = output_spec

  def output_spec(self) -> lit_types.Spec:
    return self._output_spec

  def input_spec(self) -> lit_types.Spec:
    return {}

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict]
  ) -> list[lit_types.JsonDict]:
    return []


class TCAVTest(parameterized.TestCase):

  def setUp(self):
    super(TCAVTest, self).setUp()
    self.tcav = tcav.TCAV()

  @parameterized.named_parameters(
      dict(
          testcase_name='compatible',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=True,
      ),
      dict(
          testcase_name='incompatible_align_not_in_spec',
          output_spec={
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_align_undefined',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_align_wrong_type',
          output_spec={
              'probas': lit_types.RegressionScore(),
              'embs': lit_types.Scalar(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_embeddings_not_in_spec',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_embeddings_undefined',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_embeddings_wrong_type',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Scalar(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_gradients_not_in_spec',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_gradient_target_not_in_spec',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_gradient_target_undefined',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
              ),
              'grad_target': lit_types.CategoryLabel(vocab=_TEST_VOCAB),
          },
          expected=False,
      ),
      dict(
          testcase_name='incompatible_gradient_target_wrong_type',
          output_spec={
              'probas': lit_types.MulticlassPreds(vocab=_TEST_VOCAB),
              'embs': lit_types.Embeddings(),
              'grads': lit_types.Gradients(
                  align='probas',
                  grad_for='embs',
                  grad_target_field_key='grad_target',
              ),
              'grad_target': lit_types.Scalar(),
          },
          expected=False,
      ),
  )
  def test_is_comaptible(self, output_spec: lit_types.Spec, expected: bool):
    test_model = VariableOutputSpecModel(output_spec)
    test_dataset = lit_dataset.NoneDataset({'test': test_model})
    is_compatible = self.tcav.is_compatible(test_model, test_dataset)
    self.assertEqual(is_compatible, expected)

  def test_hyp_test(self):
    # t-test where p-value != 1.
    scores = [0, 0, 0.5, 0.5, 1, 1]
    random_scores = [3, 5, -8, -100, 0, -90]
    result = self.tcav.hyp_test(scores, random_scores)
    self.assertAlmostEqual(0.1415165926492605, result)

    # t-test where p-value = 1.
    scores = [0.1, 0.13, 0.19, 0.09, 0.12, 0.1]
    random_scores = [0.1, 0.13, 0.19, 0.09, 0.12, 0.1]
    result = self.tcav.hyp_test(scores, random_scores)
    self.assertEqual(1.0, result)

  def test_compute_tcav_score(self):
    dir_deriv_positive_class = [1]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(1, result)

    dir_deriv_positive_class = [0]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(0, result)

    dir_deriv_positive_class = [1, -5, 4, 6.5, -3, -2.5, 0, 2]
    result = self.tcav.compute_tcav_score(dir_deriv_positive_class)
    self.assertAlmostEqual(0.5, result)

  def test_get_trained_cav(self):
    # 1D inputs.
    x = [[1], [1], [1], [2], [1], [1], [-1], [-1], [-2], [-1], [-1]]
    y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, 0.33, random_state=0)
    np.testing.assert_almost_equal(np.array([[19.08396947]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

    # 2D inputs.
    x = [[-8, 1], [5, 3], [3, 6], [-2, 5], [-8, 10], [10, -5]]
    y = [1, 0, 0, 1, 1, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, 0.33, random_state=0)
    np.testing.assert_almost_equal(np.array([[-77.89678676, 9.73709834]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

  def test_compute_local_scores(self):
    cav = np.array([[0, 1]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_emb': [5, 12]
        },
        {
            'probas': [0.6, 0.4],
            'cls_emb': [3, 4]
        }
    ]
    cos_sim, dot_prods = self.tcav.compute_local_scores(
        cav, dataset_outputs, 'cls_emb')
    self.assertListEqual([12, 4], dot_prods)
    # Magnitude of cav is 1, magnitude of cls_embs are [13, 5].
    # Cosine similarity is dot / (cav_mag * cls_embs_mag),
    # which is [12/13, 4/5].
    self.assertListEqual([0.9230769230769231, 0.8], cos_sim)

    cav = np.array([[1, 2, 3]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_emb': [3, 2, 1]
        },
        {
            'probas': [0.6, 0.4],
            'cls_emb': [1, 2, 0]
        }
    ]
    cos_sim, dot_prods = self.tcav.compute_local_scores(
        cav, dataset_outputs, 'cls_emb')
    self.assertListEqual([10, 5], dot_prods)
    self.assertListEqual([0.7142857142857143, 0.5976143046671968],
                         cos_sim)

  def test_get_dir_derivs(self):
    cav = np.array([[1, 2, 3]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'cls_grad': [3, 2, 1],
            'grad_class': '1'
        },
        {
            'probas': [0.6, 0.4],
            'cls_grad': [1, 2, 0],
            'grad_class': '0'
        }
    ]
    # Example where only the first output is in class_to_explain 1.
    dir_derivs = self.tcav.get_dir_derivs(
        cav, dataset_outputs, 'cls_grad', 'grad_class',
        class_to_explain='1')
    self.assertListEqual([10], dir_derivs)

if __name__ == '__main__':
  absltest.main()
