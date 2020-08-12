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
"""Tests for language.google.xnlp.citrus.utils."""
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.components.citrus import utils
import numpy as np
import tensorflow.compat.v2 as tf


class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Resets random seed for each test."""
    super().setUp()
    np.random.seed(0)

  @parameterized.named_parameters(
      {
          'testcase_name': ('Test normalization of random scores, retaining '
                            'the original sign of each value.'),
          'scores': np.random.normal(size=100),
          'make_positive': False,
      }, {
          'testcase_name': 'Test normalization of random scores.',
          'scores': np.random.normal(size=100),
          'make_positive': True,
      }, {
          'testcase_name': 'Test normalization of scores that sum to 0.',
          'scores': np.array([-2.0, 2.0]),
          'make_positive': True,
      }, {
          'testcase_name': 'Test normalization of scores are all 0.',
          'scores': np.array([0.0, 0.0]),
          'make_positive': True,
      }, {
          'testcase_name': 'Test normalization of 1-dim scores.',
          'scores': np.array([-0.5]),
          'make_positive': False,
      })
  def test_normalize_scores(self, scores, make_positive):
    """Check if the scores sum to 1 after taking their absolute values."""
    original_min = np.min(scores)
    scores = utils.normalize_scores(scores, make_positive=make_positive)
    self.assertAllClose(1.0, np.abs(scores).sum(-1))
    if not make_positive:  # Keep the sign of originally negative values.
      self.assertLessEqual(np.min(scores), np.max([0.0, original_min]))


if __name__ == '__main__':
  absltest.main()
