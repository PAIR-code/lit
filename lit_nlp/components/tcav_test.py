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
            'score':
                0.33333,
            'cos_sim': [
                0.088691, -0.12179, 0.16013, 0.24840, -0.09793, 0.05166,
                -0.21578, -0.06560, -0.14759
            ],
            'dot_prods': [
                189.085096, -266.36317, 344.350498, 547.144949, -211.663965,
                112.502439, -472.72066, -144.529598, -323.31888
            ],
            'accuracy':
                0.66667,
            'cav': [[
                0.91326976, -13.67988429, -1.33965892, 12.23702997, 4.83590182,
                -20.6276055, 19.07518597, 4.9641991, -5.52570226, 10.37834134,
                0.53685622, -5.21721739, 12.00234068, -26.36114057, 6.20088358,
                4.76729567, 26.4359945, 16.30961659, 14.70620605, -13.47771528,
                5.29218365, 0.87290488, 10.19441762, 2.96687215, -0.70745918,
                -12.201803, -10.65010904, -5.90814342, -25.17510006,
                -5.90629019, 26.53638293, 27.44054429, -11.75430314,
                -3.72779609, 10.28197421, -11.58444132, -18.09351946,
                -22.09520524, -30.04023056, -16.7551763, -12.34913637,
                30.27897114, -13.98790656, 0.65758253, 3.75770261, 4.81132118,
                -11.18005293, 5.85445303, 18.88336475, -7.34885733,
                -35.76094848, -3.39495953, 5.52774132, -8.38126488, 10.07413613,
                -1.96825956, 17.97041089, -2.47085774, 19.5700424, -49.46295186,
                0.12541183, -6.95592842, -0.33562953, 12.9269603, -13.6288284,
                -9.51211543, 21.22778867, -2.81344371, -9.66434107,
                -15.41551695, -30.15406401, 14.65903841, 2.51729041,
                17.70711379, 13.21615045, 12.55899318, 11.45749114,
                -25.67659992, 13.00876054, -1.52499005, 27.45026658,
                -4.36110401, 10.01664277, -11.24470769, -6.79411522,
                -1.67106503, -2.59389537, 11.72310319, -0.84126818, 2.03886137,
                -11.25047383, -8.60679631, -23.1676229, 22.83532544,
                -25.2657054, -15.49795527, -1.62890474, -15.49504251,
                27.26973702, -8.00652979, 6.87541861, -3.61878753, 0.82889822,
                -2.88891667, 6.13730358, 12.55884424, -2.24121286, 45.90285087,
                34.43108722, -13.32567113, 19.0988537, 2.16242269, 17.45654791,
                18.17472208, 18.28023357, -20.34869744, -11.21275755, 6.0583063,
                -10.38857432, -7.30056744, -8.85395997, -17.19779724,
                -8.12087822, 2.46058364, 29.35061395, -6.53820405, 10.3522653,
                8.54478485
            ]]
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
            'score':
                0.8,
            'cos_sim': [
                0.09527, -0.20442, 0.05141, 0.14985, 0.06750, -0.28244,
                -0.11022, -0.14479
            ],
            'dot_prods': [
                152.48776, -335.64998, 82.99588, 247.80113, 109.53684,
                -461.81805, -181.29095, -239.47817
            ],
            'accuracy':
                1.0,
            'cav': [[
                -9.18092006e+00, -3.25423105e+00, -1.95599351e+01,
                7.90544869e+00, 2.71945760e-01, 9.21655507e+00, -7.10360094e-01,
                1.06792122e+01, -4.22297728e+00, -5.88364071e+00,
                -5.90468259e+00, 3.02462186e-01, 2.73947077e+01,
                -4.21289488e+00, 4.10535370e+00, 2.90032257e+00,
                -1.15293947e+01, -6.05013625e+00, -9.63492785e+00,
                -1.24673936e+01, -7.16846202e+00, 1.65560561e+01,
                3.91797282e+00, 9.43190766e+00, 1.13459987e+01, -3.52568932e+00,
                9.13124442e-01, 9.91573080e-01, -3.25439546e+00,
                -9.93854182e-01, -1.07992122e+01, 3.11628229e+00,
                6.59499537e+00, -1.36178363e+01, 1.29858952e+01, 1.03866086e+01,
                1.59031300e+01, -5.56392889e+00, -3.77058407e+00,
                -9.32599755e+00, -9.47891226e-01, 2.05803580e+01,
                -1.37954357e+01, -7.73649342e+00, -9.07151538e+00,
                9.29099926e-01, -1.08088474e+01, -7.80737270e-01,
                2.40018316e+00, 1.09211064e+01, -3.27414948e+01,
                -2.34077057e+01, 1.17713587e+01, 1.81635854e+00, 2.31977099e+01,
                1.61744777e+00, 1.17686967e+00, -3.88545806e+00, 2.46277754e+00,
                -4.51580108e+01, 9.32061903e+00, -1.73902286e+01,
                -3.66235470e+00, 3.26925824e+01, 3.21295843e+00,
                -1.79316338e+01, 2.01011182e+01, 3.49235823e+00, 1.52195300e+01,
                -5.04835360e+00, 1.45131574e+01, 1.59716750e+01,
                -2.96747872e+00, 5.25282201e+00, -1.98576797e+01,
                -1.46852052e+00, 1.58219867e+00, -5.74221070e-01,
                1.16208072e+01, 5.11250274e-01, 5.52448443e+00, 1.11949046e+01,
                -6.93072443e-02, -7.17318663e+00, -8.48821209e+00,
                -8.71016927e+00, -1.05849366e+01, 8.71665351e+00,
                1.24408289e+01, 4.21549502e+00, -2.07208098e+01, 1.56304334e+01,
                -4.81732341e+00, 3.20792625e+01, -1.20141017e+01,
                -1.73165977e+01, -6.54369967e+00, 3.16826359e+00,
                -1.19574116e+01, -5.90525906e+00, 1.08200571e-01,
                -9.32849433e+00, -1.29775955e+00, -1.20217051e+01,
                1.30472736e+01, 2.25283409e+01, 7.04978975e-02, 5.09369848e-01,
                1.27449879e+01, -2.16621823e+00, -8.19007901e+00,
                -4.16839353e+00, 9.41498786e+00, 6.79635008e+00, 4.88519035e+00,
                2.63718274e+00, -5.28336147e-03, 1.26470921e+00,
                -3.35551546e-01, 8.26617183e+00, 2.58916933e+00,
                -8.54040920e+00, -1.18897963e+00, -5.09214401e+00,
                7.85278007e+00, -1.26074616e+01, -3.05422845e+00, 3.79508590e+00
            ]]
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
            'score':
                1.0,
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
            'accuracy':
                0.5,
            'cav': [[
                2.03451865, -13.17604027, -8.97477873, 14.2175882, 1.75463727,
                -0.46571316, 15.63798371, 9.93142109, -16.8851253, 9.11427972,
                -11.06386059, 7.19105954, -17.18586585, -4.17196936,
                29.00569093, 4.33784787, 12.33925142, -15.43126877,
                -24.48932802, 9.00534947, -2.19966647, -0.43898864, 10.17245833,
                -1.75776027, 5.32707314, -10.89541682, -2.57798881, 4.57338285,
                -1.9818453, 8.84294916, -5.4750157, 4.17024913, -19.01907373,
                -1.23878054, 0.78451806, -11.60109751, 6.45912611, 1.2580972,
                -11.36715828, 3.96784638, 5.01414527, -8.80780738, -17.95446165,
                6.28686056, 11.37040026, -9.99612493, -5.1080606, 9.22049438,
                -6.30251447, 3.63926025, 13.14918974, 14.03763789, 5.7383655,
                -8.20873775, 1.60465614, -17.91190497, -13.33602912,
                -4.49996902, 28.75901011, -3.47341192, -3.82988803, 7.55923052,
                -8.09407339, -16.11918298, -1.71507548, -1.36150357,
                -5.86025439, 15.68441846, -3.51353798, 2.43790252, -2.19348517,
                5.0211597, 2.37646679, 13.41654008, 0.55570348, -6.05668506,
                -8.42704919, -5.02851713, 14.84420539, -7.34185174, 4.03091355,
                -5.09645307, 8.2008516, 1.08925377, -0.58413633, 9.92903691,
                2.14726806, 13.30179969, 13.57605882, -2.90356308, -5.45706524,
                10.79901632, -7.80617599, 17.51044709, -7.5189762, -16.52918401,
                -0.52797854, -2.44961364, 10.13560444, -13.19007029,
                -0.93666233, 0.54694589, 1.92814453, -4.7547722, -1.39708064,
                6.24580764, -1.41235513, -2.47243578, 10.31137604, -11.16312467,
                -10.42280945, -13.08010871, 17.36711388, 5.96436519,
                -11.97783485, 18.54090287, 7.91701915, -17.52516079,
                -13.20368127, -26.3674698, -2.45510052, 5.56414756, 10.80934702,
                -8.21923153, -11.8954632, -4.44263104, -3.42691711, -4.79007185
            ]]
        },
        'p_val': None,
        'random_mean': 0.9285714285714286,
        'split_size': 3,
        'num_runs': 1
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0], places=3)

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
            'score':
                0.0,
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
            'accuracy':
                0.5,
            'cav': [[
                7.55329281, -2.62460237, -9.0024543, 0.59818666, -14.58093322,
                -15.42218557, -5.5354822, 12.364417, 21.63543734, -12.10592188,
                -28.13137179, 14.31976269, -31.59852801, -9.5688074,
                12.95670217, -22.25037747, 16.35172203, 1.05301289, -31.3269959,
                -2.59048782, 19.77513966, 10.23254963, 4.18399631, 4.87700827,
                -16.33114267, 2.79754082, 10.54735581, 11.78229815,
                -12.25723096, 4.70124727, -11.24257086, -5.43199386,
                -7.69344958, 14.56370698, -5.75688441, -5.26137455, 23.44984724,
                -6.31077994, 17.08448569, 9.05777822, 1.6895136, -17.90790926,
                -14.34180658, 10.53637, 20.52315338, 13.25800162, 10.93816225,
                -3.75636305, -7.45849454, -3.97231903, -2.56363646, 7.11917039,
                8.16617295, -7.89912694, 12.99981985, -21.34898036,
                -17.63337856, -2.63946913, -4.40439706, 6.52652447, 3.80778434,
                -9.19669592, -6.7804855, -1.69738751, 14.07850725, -19.72428768,
                -4.39987548, 0.52300402, -1.32919268, -1.17373043, 25.24466652,
                1.10433416, 2.28887964, 2.73226767, 11.38988286, -15.36477422,
                1.72328432, -10.98681852, 4.09940512, 1.35212746, -11.08208377,
                15.02775212, 24.25939889, 2.5819625, -4.82504369, -2.8454034,
                24.2562686, -4.11429068, -1.74594319, -19.37503033, 2.29546159,
                6.19942707, 3.62587758, 17.79947087, 1.53031934, 2.38564874,
                -16.59565939, -5.20520209, -4.84197479, -9.78838634,
                -12.3678921, 16.95331818, 7.66526629, 20.39393112, 2.87282507,
                -17.29492882, -7.67759675, 2.78484882, 7.79825779, -5.36932978,
                12.74328979, -5.22601154, 3.36537017, -16.2269691, 5.08010908,
                8.33516097, 6.93697789, -30.79130942, 8.65766065, 14.05914846,
                1.90387642, 7.06612428, 13.16682386, -7.97255194, -15.33183742,
                2.61200359, -2.05912859, 10.68784471
            ]]
        },
        'p_val': 0.42264973081037427,
        'random_mean': 0.0,
        'split_size': 3,
        'num_runs': 2
    }

    testing_utils.assert_deep_almost_equal(self, expected, result[0], places=3)

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
            'score':
                0.42857142857142855,
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
            'accuracy':
                0.75,
            'cav': [[
                -2.01128835e+01, -2.53418963e+01, 3.02074307e+01,
                3.78726319e+00, -6.30370696e+00, 1.74700275e+00, 2.26769890e+01,
                -1.06487415e+01, -1.76710295e+00, 1.53911930e+01,
                6.54822833e+00, 2.02141240e+00, 1.16919236e+00, -4.59410912e-01,
                4.54187735e+00, 1.98659207e+01, 1.01712167e-01, 1.18509678e+01,
                3.11218189e+01, 1.60491967e+01, -2.84655620e+01,
                -1.08942472e+01, -1.64852881e+01, 4.79670684e+00,
                3.08027345e+01, -2.64932823e+01, 1.14056950e+00,
                -1.45201048e+01, -8.41149151e+00, -1.26598740e+01,
                -5.87020201e+00, -2.87027220e+00, 2.13724773e+01,
                3.16488669e+01, -1.38671518e+01, 2.25443206e+01,
                -1.93658542e+01, 1.47272427e+01, 1.42369662e+01,
                -7.98238134e+00, -2.34073887e+01, 3.63425744e+00,
                3.20426643e+00, -1.57109214e+01, -1.87085664e+01,
                2.04525909e-02, 1.75499187e+01, -1.89476242e+00, 1.22602038e+01,
                -8.30212717e-01, 6.89138456e+00, -4.76848738e+01,
                -1.02832284e+01, 1.25879647e+01, -2.34436812e+01,
                3.46227988e+01, 3.98466501e+00, -4.18899781e+01, 2.31382829e+01,
                9.44899112e+00, -5.55529692e+00, 2.99535098e+01, 1.91656336e+01,
                1.90906964e+01, -1.97222943e+01, -8.81455291e+00,
                -4.31148519e-01, 1.35055378e+01, -2.93433908e+01,
                1.41053039e+01, 7.31250277e+00, 1.84754657e+01, -3.59438637e+00,
                -1.94830172e+01, 1.82690513e+00, 1.11006786e+01,
                -1.98780467e+01, 9.70483509e+00, 3.10464830e+01,
                -1.58164540e+00, 3.92983591e+01, -1.94148650e+01,
                -1.78218017e+01, 1.68395976e+01, 1.88509800e+01,
                -4.67428605e+00, -4.11605219e+01, 4.20027328e+00,
                -1.78356055e+01, 2.93033407e+01, 3.67970865e+00, 4.43803533e+00,
                -5.86478123e+00, -1.89515809e+01, -4.15883069e+00,
                -2.53725359e+01, -2.06732149e+01, 3.51060797e+00,
                7.32087133e+00, 7.98146613e-01, 3.38068707e+00, -2.48928693e+01,
                1.99907796e+00, 2.12458621e+00, -1.33852676e+01, 1.42904487e+01,
                1.33286404e-02, -3.82017252e+00, -1.04635257e+01,
                1.00280360e+01, 8.93273804e+00, 5.26520675e+00, -1.28270015e+01,
                3.54992057e+00, 2.32608393e+00, -1.84115520e+01,
                -1.74357882e+00, 2.53797871e+01, -1.75258907e+01,
                -4.22843710e+01, -1.49635329e+01, -3.77852872e+00,
                -1.25404291e+01, 1.17741455e+01, 3.35462077e+01, 1.65145075e+01,
                4.27775903e-01, 8.26403983e+00
            ]]
        },
        'p_val': 0.04400624968940752,
        'random_mean': 0.9642857142857143,
        'split_size': 5,
        'num_runs': 2
    }]
    testing_utils.assert_deep_almost_equal(self, expected, result, places=3)


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
