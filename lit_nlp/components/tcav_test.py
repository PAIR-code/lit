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
from lit_nlp.lib import caching  # for hash id fn
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
    self.model = caching.CachingModelWrapper(
        glue_models.SST2Model(BERT_TINY_PATH), 'test')

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
        {'sentence': 'h'},
        {'sentence': 'i'}]

    indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                      for ex in examples]
    dataset = lit_dataset.IndexedDataset(id_fn=caching.input_hash,
                                         indexed_examples=indexed_inputs)
    config = {
        'concept_set_ids': [indexed_inputs[0]['id'],
                            indexed_inputs[2]['id'],
                            indexed_inputs[3]['id'],
                            indexed_inputs[7]['id']],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0,
        'dataset_name': 'test'
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

    indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                      for ex in examples]
    dataset = lit_dataset.IndexedDataset(id_fn=caching.input_hash,
                                         indexed_examples=indexed_inputs)
    config = {
        'concept_set_ids': [indexed_inputs[0]['id'],
                            indexed_inputs[2]['id'],
                            indexed_inputs[3]['id'],
                            indexed_inputs[4]['id'],
                            indexed_inputs[7]['id']],
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

  def test_relative_tcav(self):
    # Tests passing in a negative set.

    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    examples = [
        {'sentence': 'happy'},  # 0
        {'sentence': 'sad'},  # 1
        {'sentence': 'good'},  # 2
        {'sentence': 'bad'},  # 3
        {'sentence': 'pretty'},  # 4
        {'sentence': 'ugly'},  # 5
        {'sentence': 'sweet'},  # 6
        {'sentence': 'bitter'},  # 7
        {'sentence': 'well'},  # 8
        {'sentence': 'poor'},  # 9
        {'sentence': 'compelling'},  # 10
        {'sentence': 'boring'},  # 11
        {'sentence': 'pleasing'},  # 12
        {'sentence': 'gross'},  # 13
        {'sentence': 'blue'},  # 14
        {'sentence': 'red'},  # 15
        {'sentence': 'flower'},  # 16
        {'sentence': 'bee'},  # 17
        {'sentence': 'snake'},  # 18
        {'sentence': 'windshield'},  # 19
        {'sentence': 'plant'},  # 20
        {'sentence': 'scary'},  # 21
        {'sentence': 'pencil'},  # 22
        {'sentence': 'hello'}  # 23
    ]

    indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                      for ex in examples]
    dataset = lit_dataset.IndexedDataset(id_fn=caching.input_hash,
                                         indexed_examples=indexed_inputs)

    # This first example doesn't have enough examples for statistical testing,
    # so the returned p-value is None.
    config = {
        'concept_set_ids': [indexed_inputs[0]['id'],
                            indexed_inputs[2]['id'],
                            indexed_inputs[4]['id']],
        'negative_set_ids': [indexed_inputs[1]['id'],
                             indexed_inputs[3]['id'],
                             indexed_inputs[5]['id']],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }

    result = self.tcav.run_with_metadata(indexed_inputs, self.model, dataset,
                                         config=config)

    self.assertLen(result, 1)
    expected = {
        'result': {
            'score': 1.0,
            'cos_sim': [
                0.9999999581246426, 0.049332143689572144, 0.8987945047547466,
                -0.41858423757857954, 0.6908297036543664, -0.5167857909664919,
                0.8423017503220364, -0.005793079244916016, 0.8334491603894322,
                -0.4054645113448612, 0.7616102123736647, -0.4578596155267783,
                0.8366905563807711, -0.27390786544756535, 0.7325538474066896,
                0.5190287630768531, 0.8145227936096425, 0.02005592868363552,
                -0.1143256029298114, -0.1221480700842533, 0.6852995739227957,
                0.3984620730733816, 0.5211149530112407, 0.5909723902471223
            ],
            'dot_prods': [
                1385.1480610241554, 69.95638452724207, 1239.4947646060161,
                -595.253135700978, 971.5880156862692, -725.0749813217176,
                1182.8641913758102, -8.149647641120662, 1146.5803071544124,
                -576.4043054391316, 1038.3510704649307, -648.097269442522,
                1154.4720122394317, -378.32103870822493, 1024.066390571124,
                738.6959135414066, 1139.7963358416857, 28.691395032352318,
                -167.37808507284706, -176.4474746971391, 959.5159619261449,
                562.8772536987927, 716.7270332848395, 840.7031847912738
            ],
            'accuracy': 0.5
        },
        'p_val': None,
        'random_mean': 0.9285714285714286,
        'split_size': 3,
        'num_runs': 1
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0])

    # This example has enough inputs for two runs of size 3.
    config = {
        'concept_set_ids': [
            indexed_inputs[1]['id'], indexed_inputs[2]['id'],
            indexed_inputs[4]['id'], indexed_inputs[5]['id'],
            indexed_inputs[10]['id'], indexed_inputs[9]['id']
        ],
        'negative_set_ids': [
            indexed_inputs[0]['id'], indexed_inputs[3]['id'],
            indexed_inputs[12]['id'], indexed_inputs[6]['id'],
            indexed_inputs[7]['id'], indexed_inputs[8]['id']
        ],
        'class_to_explain': '0',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }

    result = self.tcav.run_with_metadata(indexed_inputs, self.model, dataset,
                                         config=config)
    self.assertLen(result, 1)
    expected = {
        'result': {
            'score': 0.0,
            'cos_sim': [
                0.2731987606830683, 0.427838045403812, 0.3166440584420665,
                -0.1358964965831398, 0.5616614702946262, -0.16511808390168164,
                -0.05103355252438478, -0.16945565920473257, 0.28148962348967155,
                -0.18169036476392003, 0.33244873698665106, -0.13316476546155087,
                0.15226772288202886, -0.05534469666649352, 0.2886150002073456,
                0.33888135113008555, 0.12875301375254147, 0.046908665182593096,
                -0.052445114502024985, 0.088858405172313, 0.219517174438115,
                0.35833013079793435, 0.2291162415605806, 0.3635686086637199
            ],
            'dot_prods': [
                452.17220644153525, 724.9460578876271, 521.776546745851,
                -230.9170522777958, 943.8754747127095, -276.8190148523963,
                -85.63511897570154, -284.8487792023684, 462.71830216201926,
                -308.62790255581496, 541.5830529968077, -225.2299308998058,
                251.04716264718752, -91.33998249705493, 482.0991668852444,
                576.3029773313335, 215.28329927312336, 80.18458502795752,
                -91.74640483442752, 153.37559992294862, 367.2562273288043,
                604.8378479001944, 376.53473821563625, 618.003311205616
            ],
            'accuracy': 0.5
        },
        'p_val': 0.42264973081037427,
        'random_mean': 0.0,
        'split_size': 3,
        'num_runs': 2
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0])

    # This example has enough examples for three runs of size 3 and two runs of
    # size 5, and returns results with p-value < 0.05.
    config = {
        'concept_set_ids': [indexed_inputs[0]['id'],
                            indexed_inputs[1]['id'],
                            indexed_inputs[2]['id'],
                            indexed_inputs[3]['id'],
                            indexed_inputs[4]['id'],
                            indexed_inputs[5]['id'],
                            indexed_inputs[6]['id'],
                            indexed_inputs[7]['id'],
                            indexed_inputs[8]['id'],
                            indexed_inputs[9]['id']],
        'negative_set_ids': [indexed_inputs[10]['id'],
                             indexed_inputs[11]['id'],
                             indexed_inputs[12]['id'],
                             indexed_inputs[13]['id'],
                             indexed_inputs[14]['id'],
                             indexed_inputs[15]['id'],
                             indexed_inputs[16]['id'],
                             indexed_inputs[17]['id'],
                             indexed_inputs[18]['id'],
                             indexed_inputs[19]['id']],
        'class_to_explain': '1',
        'grad_layer': 'cls_grad',
        'random_state': 0
    }

    result = self.tcav.run_with_metadata(indexed_inputs, self.model, dataset,
                                         config=config)
    self.assertLen(result, 1)
    expected = [{
        'result': {
            'score': 0.42857142857142855,
            'cos_sim': [
                -0.1107393877916321, -0.0993967046974328, -0.2214985917242054,
                0.08132588965575606, -0.3590211572508748, 0.18708109817461333,
                0.000724498781128839, 0.09700473783330398, -0.25015742815240055,
                0.16108236033785076, -0.10283274286140846, 0.0972663321478731,
                -0.05924679176256152, -0.048499696342091746,
                -0.4357117016074766, -0.593245752003111, -0.3645147796989344,
                -0.5507605083253673, -0.27914997949782694, -0.30908550968594417,
                -0.5584676299422896, -0.16983339994284577, -0.42587740852240746,
                -0.37482298817032594
            ],
            'dot_prods': [
                -261.4389298435066, -240.23776409902007, -520.6275907607769,
                197.11495117497446, -860.6035066083074, 447.3775519523981,
                1.7341104803878409, 232.59170976304426, -586.5576327736542,
                390.2961568516803, -238.95427152619726, 234.6617547723058,
                -139.3334215524385, -114.17392512371171, -1038.149036709951,
                -1439.0663895591745, -869.3828698612926, -1342.899780229334,
                -696.569760699206, -760.9907977738051, -1332.7284530349625,
                -408.90435403478875, -998.3360993150825, -908.8111404537224
            ],
            'accuracy': 0.75
        },
        'p_val': 0.04400624968940752,
        'random_mean': 0.9642857142857143,
        'split_size': 5,
        'num_runs': 2
    }]
    testing_utils.assert_deep_almost_equal(self, expected, result)


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
