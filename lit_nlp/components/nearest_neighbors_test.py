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

from typing import List

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import nearest_neighbors
from lit_nlp.lib import caching  # for hash id fn
from lit_nlp.lib import testing_utils
import numpy as np


JsonDict = lit_types.JsonDict


class TestModelNearestNeighbors(lit_model.Model):
  """Implements lit.Model interface for nearest neighbors.

     Returns the same output for every input.
  """

  # LIT API implementation
  def max_minibatch_size(self, **unused_kw):
    return 3

  def input_spec(self):
    return {'segment': lit_types.TextSegment}

  def output_spec(self):
    return {'probas': lit_types.MulticlassPreds(
        parent='label',
        vocab=['0', '1'],
        null_idx=0),
            'input_embs': lit_types.TokenEmbeddings(align='tokens'),
            }

  def predict_minibatch(self, inputs: List[JsonDict], **kw):
    embs = [np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 0]),
            np.array([5, 8, -10, 0])]
    probas = np.array([0.2, 0.8])
    return [{'probas': probas, 'input_embs': embs[i]}
            for i, _ in enumerate(inputs)]


class NearestNeighborTest(absltest.TestCase):

  def setUp(self):
    super(NearestNeighborTest, self).setUp()
    self.nearest_neighbors = nearest_neighbors.NearestNeighbors()

  def test_run_nn(self):
    examples = [
        {
            'segment': 'a'
        },
        {
            'segment': 'b'
        },
        {
            'segment': 'c'
        },
    ]
    indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                      for ex in examples]

    model = TestModelNearestNeighbors()
    dataset = lit_dataset.IndexedDataset(id_fn=caching.input_hash,
                                         indexed_examples=indexed_inputs)
    config = {
        'embedding_name': 'input_embs',
        'num_neighbors': 2,
    }
    result = self.nearest_neighbors.run_with_metadata([indexed_inputs[1]],
                                                      model, dataset,
                                                      config=config)
    expected = {'nearest_neighbors': [
        {'id': '1', 'nn_distance': 0.0},
        {'id': '0', 'nn_distance': 1.7320508075688772}]}

    self.assertLen(result, 1)
    testing_utils.assert_deep_almost_equal(self, expected, result[0])

if __name__ == '__main__':
  absltest.main()
