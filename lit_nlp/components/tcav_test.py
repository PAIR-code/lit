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
"""Tests for lit_nlp.components.gradient_maps."""

import random

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import tcav
# TODO(lit-dev): Move glue_models out of lit_nlp/examples
from lit_nlp.examples.models import glue_models
from lit_nlp.lib import testing_utils
import numpy as np


JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
extract_compressed_file=True)


class ModelBasedTCAVTest(absltest.TestCase):

  def setUp(self):
    super(ModelBasedTCAVTest, self).setUp()
    self.tcav = tcav.TCAV()
    self.model = glue_models.SST2Model(BERT_TINY_PATH)

  def test_tcav(self):
    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    examples = [
        {'sentence': 'a'},
        {'sentence': 'b'},
        {'sentence': 'c'},
        {'sentence': 'd'},
        {'sentence': 'e'},
        {'sentence': 'f'},
        {'sentence': 'g'},
        {'sentence': 'h'}]
    indexed_inputs = [
        {
            'id': '1',
            'data': {
                'sentence': 'a'
            }
        },
        {
            'id': '2',
            'data': {
                'sentence': 'b'
            }
        },
        {
            'id': '3',
            'data': {
                'sentence': 'c'
            }
        },
        {
            'id': '4',
            'data': {
                'sentence': 'd'
            }
        },
        {
            'id': '5',
            'data': {
                'sentence': 'e'
            }
        },
        {
            'id': '6',
            'data': {
                'sentence': 'f'
            }
        },
        {
            'id': '7',
            'data': {
                'sentence': 'g'
            }
        },
        {
            'id': '8',
            'data': {
                'sentence': 'h'
            }
        },
        {
            'id': '9',
            'data': {
                'sentence': 'i'
            }
        },
    ]

    dataset_spec = {'sentence': lit_types.TextSegment()}
    dataset = lit_dataset.Dataset(dataset_spec, examples)
    config = {
        'concept_set_ids': ['1', '3', '4', '8'],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }
    result = self.tcav.run_with_metadata(indexed_inputs, self.model, dataset,
                                         config=config)

    self.assertLen(result, 1)
    expected = {
        'p_val': 0.13311,
        'random_mean': 0.56667,
        'result': {
            'score': 0.33333,
            'cos_sim': [
                0.088691, -0.12179, 0.16013,
                0.24840, -0.09793, 0.05166,
                -0.21578, -0.06560, -0.14759
            ],
            'dot_prods': [
                189.085096, -266.36317, 344.350498,
                547.144949, -211.663965, 112.502439,
                -472.72066, -144.529598, -323.31888
            ],
            'accuracy': 0.66667
        }
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0])

  def test_tcav_sample_from_positive(self):
    # Tests the case where more concept examples are passed than non-concept
    # examples, so the concept set is sampled from the concept examples.

    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    examples = [
        {'sentence': 'a'},
        {'sentence': 'b'},
        {'sentence': 'c'},
        {'sentence': 'd'},
        {'sentence': 'e'},
        {'sentence': 'f'},
        {'sentence': 'g'},
        {'sentence': 'h'}]
    indexed_inputs = [
        {
            'id': '1',
            'data': {
                'sentence': 'a'
            }
        },
        {
            'id': '2',
            'data': {
                'sentence': 'b'
            }
        },
        {
            'id': '3',
            'data': {
                'sentence': 'c'
            }
        },
        {
            'id': '4',
            'data': {
                'sentence': 'd'
            }
        },
        {
            'id': '5',
            'data': {
                'sentence': 'e'
            }
        },
        {
            'id': '6',
            'data': {
                'sentence': 'f'
            }
        },
        {
            'id': '7',
            'data': {
                'sentence': 'g'
            }
        },
        {
            'id': '8',
            'data': {
                'sentence': 'h'
            }
        },
    ]
    dataset_spec = {'sentence': lit_types.TextSegment()}
    dataset = lit_dataset.Dataset(dataset_spec, examples)
    config = {
        'concept_set_ids': ['1', '3', '4', '5', '8'],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }
    result = self.tcav.run_with_metadata(indexed_inputs, self.model, dataset,
                                         config=config)

    self.assertLen(result, 1)
    expected = {
        'p_val': 0.80489,
        'random_mean': 0.53333,
        'result': {
            'score': 0.8,
            'cos_sim': [
                0.09527, -0.20442, 0.05141,
                0.14985, 0.06750, -0.28244,
                -0.11022, -0.14479
            ],
            'dot_prods': [
                152.48776, -335.64998, 82.99588,
                247.80113, 109.53684, -461.81805,
                -181.29095, -239.47817
            ],
            'accuracy': 1.0
        }
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0])


class TCAVTest(absltest.TestCase):

  def setUp(self):
    super(TCAVTest, self).setUp()
    self.tcav = tcav.TCAV()
    self.model = glue_models.SST2Model(BERT_TINY_PATH)

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
