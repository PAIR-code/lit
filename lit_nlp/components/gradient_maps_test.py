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
from lit_nlp.components import gradient_maps
from lit_nlp.lib import testing_utils
import numpy as np

CLASS_KEY = gradient_maps.CLASS_KEY
INTERPOLATION_KEY = gradient_maps.INTERPOLATION_KEY
NORMALIZATION_KEY = gradient_maps.NORMALIZATION_KEY


class GradientMapsTest(parameterized.TestCase):

  def setUp(self):
    super(GradientMapsTest, self).setUp()
    self.ig = gradient_maps.IntegratedGradients()

  # Integrated gradients tests
  def test_gradient_maps(self):
    self.ig = gradient_maps.IntegratedGradients()

    # Basic test with dummy outputs from the model.
    inputs = [{'segment': '_'}]
    model = testing_utils.TestModelClassification()
    dataset = lit_dataset.Dataset(None, None)
    output = self.ig.run(inputs, model, dataset)

    self.assertLen(output, 1)

    salience = output[0]['input_embs_grad'].salience
    target = np.array([0.25, 0.25, 0.25, 0.25])
    self.assertTrue((salience == target).all())

  def test_get_baseline(self):
    self.ig = gradient_maps.IntegratedGradients()
    result = self.ig.get_baseline(
        np.array([[[1, 1, 1, 1], [2, 3, 6, 7], [3, 4, 5, 6]]]))
    target = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    np.testing.assert_almost_equal(result, target)

  def test_estimate_integral(self):
    self.ig = gradient_maps.IntegratedGradients()

    result = self.ig.estimate_integral(np.array([[0], [1], [2]]))
    target = np.array([1])
    np.testing.assert_almost_equal(result, target)

    result = self.ig.estimate_integral(np.array([[0, 0], [0, 0], [0, 1]]))
    target = np.array([0, 0.25])
    np.testing.assert_almost_equal(result, target)

  def test_get_interpolated_inputs(self):
    self.ig = gradient_maps.IntegratedGradients()
    result = self.ig.get_interpolated_inputs(np.array([[0], [1]]),
                                             np.array([[1], [0]]), 2)
    target = np.array([[[0], [1]], [[0.5], [0.5]], [[1], [0]]])
    np.testing.assert_almost_equal(result, target)
    np.testing.assert_almost_equal(result, target)

  @parameterized.named_parameters(
      ('default', None, None, None, None),
      ('autorun only', True, None, None, None),
      ('config values only', None, 'test_class_key', 50, False),
      ('autorun + config values', True, 'test_class_key', 50, False))
  def test_ig_init_args(self, autorun, class_key, interpolation_steps,
                        normalize):
    params = {}
    if autorun is not None:
      params['autorun'] = autorun
    if class_key is not None:
      params['class_key'] = class_key
    if interpolation_steps is not None:
      params['interpolation_steps'] = interpolation_steps
    if normalize is not None:
      params['normalize'] = normalize

    ig = gradient_maps.IntegratedGradients(**params)
    config_spec = ig.config_spec()

    if autorun is not None:
      self.assertEqual(ig.meta_spec()['saliency'].autorun, autorun)
    if class_key is not None:
      self.assertEqual(config_spec[CLASS_KEY].default, class_key)
    if interpolation_steps is not None:
      self.assertEqual(config_spec[INTERPOLATION_KEY].default,
                       interpolation_steps)
    if normalize is not None:
      self.assertEqual(config_spec[NORMALIZATION_KEY].default, normalize)


if __name__ == '__main__':
  absltest.main()
