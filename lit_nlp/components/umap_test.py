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
"""Tests for lit_nlp.components.umap."""
from absl.testing import absltest
from lit_nlp.components import umap
from lit_nlp.lib import testing_utils
import numpy as np


class UmapTest(absltest.TestCase):

  def test_fit_transform(self):
    umap_model = umap.UmapModel(n_components=3)

    # Make fake embeddings.
    n = 100
    inputs = testing_utils.fake_projection_input(n, 10)

    outputs = umap_model.fit_transform(inputs)
    outputs_list = list(outputs)

    # Check that the output dict keys are correct.
    self.assertIn('z', outputs_list[0])

    # Check that the _fitted flag has been flipped.
    self.assertTrue(umap_model._fitted)

    # Check that the output shape is correct.
    output_np = np.array([o['z'] for o in outputs_list])
    shape = output_np.shape
    expected_shape = (n, 3)
    self.assertEqual(shape, expected_shape)

  def test_predict_minibatch(self):
    umap_model = umap.UmapModel(n_components=3)

    # Test falsy return value when umap hasn't been initialized.
    num_dims = 10
    inputs = testing_utils.fake_projection_input(1, num_dims)
    output = umap_model.predict_minibatch(inputs)
    self.assertEqual(list(output)[0]['z'], [0, 0, 0])

    # Make dummy embeddings to warm start.
    umap_model.fit_transform(testing_utils.fake_projection_input(100, num_dims))

    # Test that we can now predict a minibatch.
    output = umap_model.predict_minibatch(inputs)
    output_shape = np.array(list(output)[0]['z']).shape
    self.assertEqual(output_shape, (3,))


if __name__ == '__main__':
  absltest.main()
