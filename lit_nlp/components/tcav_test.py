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
from lit_nlp.components import tcav
from lit_nlp.lib import testing_utils
import numpy as np


class TCAVTest(absltest.TestCase):

  def setUp(self):
    super(TCAVTest, self).setUp()
    self.tcav = tcav.TCAV()

  def test_create_comparison_splits(self):
    random.seed(0)

    examples = [
        {'segment': 'a'},
        {'segment': 'b'},
        {'segment': 'c'},
        {'segment': 'd'},
    ]
    dataset = lit_dataset.Dataset(None, examples)
    concept_set = [examples[1]]
    # Generates 3 splits of size 1.
    result = self.tcav.create_comparison_splits(dataset, concept_set,
                                                num_splits=3)
    expected = [[{'segment': 'd'}], [{'segment': 'd'}], [{'segment': 'a'}]]
    self.assertListEqual(expected, result)

  def test_hyp_test(self):
    # t-test where p-value = 1.
    scores = [0, 0, 0.5, 0.5, 1, 1]
    result = self.tcav.hyp_test(scores)
    self.assertEqual(1, result)

    # t-test where p-value ~ 0.
    scores = [0.1, 0.13, 0.19, 0.09, 0.12, 0.1]
    result = self.tcav.hyp_test(scores)
    self.assertAlmostEqual(1.7840024559935266e-06, result)

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

  def test_tcav(self):
    random.seed(0)  # Sets seed since create_comparison_splits() uses random.

    # Basic test with dummy outputs from the model.
    inputs = [
        {'segment': 'a'},
        {'segment': 'c'},
        {'segment': 'd'},
        {'segment': 'h'},
        {'segment': 'f'},
        {'segment': 'h'}]
    examples = [
        {'segment': 'a'},
        {'segment': 'b'},
        {'segment': 'c'},
        {'segment': 'd'},
        {'segment': 'e'},
        {'segment': 'f'},
        {'segment': 'g'},
        {'segment': 'h'},
    ]
    model = testing_utils.TestModelClassification()
    dataset = lit_dataset.Dataset(None, examples)
    config = {
        'concepts': {
            'concept_1': (0, 4),
            'concept_2': (4, 6)
        },
        'gradient_class': 1,
        'layer': 'embedding',
        'random_state': 0
    }
    result = self.tcav.run(inputs, model, dataset, config=config)

    self.assertLen(result, 1)
    expected = {
        'concept_1': {
            'result': {
                'score': 0.0,
                'dir_derivs': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'accuracy': 0.3333333333333333
            },
            'p_val': 0.0
        },
        'concept_2': {
            'result': {
                'score': 0.0,
                'dir_derivs': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'accuracy': 0.5
            },
            'p_val': 0.0
        }
    }
    self.assertDictEqual(expected, result[0])

  def test_get_trained_cav(self):
    # 1D inputs.
    x = [[1], [1], [1], [2], [1], [1], [-1], [-1], [-2], [-1], [-1]]
    y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, random_state=0)
    np.testing.assert_almost_equal(np.array([[19.08396947]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

    # 2D inputs.
    x = [[-8, 1], [5, 3], [3, 6], [-2, 5], [-8, 10], [10, -5]]
    y = [1, 0, 0, 1, 1, 0]
    cav, accuracy = self.tcav.get_trained_cav(x, y, random_state=0)
    np.testing.assert_almost_equal(np.array([[-77.89678676, 9.73709834]]), cav)
    self.assertAlmostEqual(1.0, accuracy)

  def test_get_dir_derivs(self):
    cav = np.array([[1, 2, 3]])
    dataset_outputs = [
        {
            'probas': [0.2, 0.8],
            'embedding_cls_grad': [3, 2, 1]
        },
        {
            'probas': [0.6, 0.4],
            'embedding_cls_grad': [1, 2, 0]
        }
    ]
    # Example where only the first output is in gradient_class 1.
    dir_derivs, dir_derivs_positive_class = self.tcav.get_dir_derivs(
        cav, dataset_outputs, 'embedding', gradient_class=1)
    self.assertListEqual([10, 5], dir_derivs)
    self.assertListEqual([10], dir_derivs_positive_class)

if __name__ == '__main__':
  absltest.main()
