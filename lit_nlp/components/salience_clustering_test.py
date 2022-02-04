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

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.components import gradient_maps
from lit_nlp.components import salience_clustering
from lit_nlp.lib import testing_utils
import numpy as np


class SalienceClusteringTest(absltest.TestCase):

  def setUp(self):
    super(SalienceClusteringTest, self).setUp()
    self.salience_mappers = {
        'Grad L2 Norm': gradient_maps.GradientNorm(),
        'Grad ⋅ Input': gradient_maps.GradientDotInput()
    }

  def test_build_vocab(self):
    token_saliencies = [
        {
            'token_grad_sentence':
                dtypes.TokenSalience(
                    ['a', 'b', 'c'],
                    np.array([0, 0, 0]),
                )
        },
        {
            'token_grad_sentence':
                dtypes.TokenSalience(
                    ['d', 'e', 'f'],
                    np.array([0, 0, 0]),
                )
        },
    ]

    clustering_component = salience_clustering.SalienceClustering(
        self.salience_mappers)
    vocab_lookup, vocab = clustering_component._build_vocab(token_saliencies)
    expected_vocab_lookup = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    expected_vocab = ['a', 'b', 'c', 'd', 'e', 'f']
    self.assertEqual(expected_vocab_lookup, vocab_lookup)
    self.assertEqual(expected_vocab, vocab)

  def test_convert_to_bow_vector(self):
    token_saliencies = [
        {
            'token_grad_sentence':
                dtypes.TokenSalience(
                    ['a', 'b', 'c'],
                    np.array([0.1, 0.2, 0.3]),
                )
        },
        # Checks that for 2 equal tokens the one with the largest absolute value
        # is preserved.
        {
            'token_grad_sentence':
                dtypes.TokenSalience(
                    ['d', 'e', 'd'],
                    np.array([0.4, 0.5, -0.6]),
                )
        },
    ]

    clustering_component = salience_clustering.SalienceClustering(
        self.salience_mappers)
    vocab_lookup = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    representations = clustering_component._compute_fixed_length_representation(
        token_saliencies, vocab_lookup)
    expected = [
        {
            'token_grad_sentence':
                np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]) / np.sqrt(0.14)
        },
        {
            'token_grad_sentence':
                np.array([0.0, 0.0, 0.0, -0.6, 0.5, 0.0]) / np.sqrt(0.61)
        },
    ]
    np.testing.assert_equal(expected, representations)

  def test_clustering(self):
    inputs = [
        {
            'data': {
                'segment': 'a b c d'
            }
        },
        {
            'data': {
                'segment': 'a b c d'
            }
        },
        {
            'data': {
                'segment': 'e f e f'
            }
        },
        {
            'data': {
                'segment': 'e f e f'
            }
        },
        {
            'data': {
                'segment': 'e f e f'
            }
        },
    ]
    model = testing_utils.TestModelClassification()
    dataset = lit_dataset.Dataset(None, None)
    config = {
        salience_clustering.SALIENCE_MAPPER_KEY: 'Grad L2 Norm',
        salience_clustering.N_CLUSTERS_KEY: 2,
        salience_clustering.N_TOP_TOKENS_KEY: 2
    }

    model_outputs = [{
        'input_embs_grad':
            np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1]]),
        'tokens': ['a', 'b', 'c', 'd'],
        'grad_class':
            '1'
    }, {
        'input_embs_grad':
            np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 1]]),
        'tokens': ['a', 'b', 'c', 'd'],
        'grad_class':
            '1'
    }, {
        'input_embs_grad':
            np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
        'tokens': ['e', 'f', 'e', 'g'],
        'grad_class':
            '1'
    }, {
        'input_embs_grad':
            np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
        'tokens': ['e', 'f', 'e', 'g'],
        'grad_class':
            '1'
    }, {
        'input_embs_grad':
            np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]),
        'tokens': ['e', 'f', 'e', 'g'],
        'grad_class':
            '1'
    }]

    clustering_component = salience_clustering.SalienceClustering(
        self.salience_mappers)
    result = clustering_component.run_with_metadata(inputs, model, dataset,
                                                    model_outputs, config)
    # Cluster id assignment is random, so in one run the first 2 examples may
    # be cluster 0, in the next run they may be in cluster 1.
    cluster_id_of_first = result[
        salience_clustering.CLUSTER_ID_KEY]['input_embs_grad'][0]
    cluster_id_of_last = result[
        salience_clustering.CLUSTER_ID_KEY]['input_embs_grad'][-1]
    np.testing.assert_equal(
        result[salience_clustering.CLUSTER_ID_KEY]['input_embs_grad'], [
            cluster_id_of_first, cluster_id_of_first, cluster_id_of_last,
            cluster_id_of_last, cluster_id_of_last
        ])
    np.testing.assert_allclose(
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][0],
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][1])
    np.testing.assert_allclose(
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][2],
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][3])
    np.testing.assert_allclose(
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][2],
        result[salience_clustering.REPRESENTATION_KEY]['input_embs_grad'][4])
    self.assertIn('input_embs_grad', clustering_component.kmeans)
    self.assertIsNotNone(clustering_component.kmeans['input_embs_grad'])

    # Clustering isn't deterministic so we don't know if examples 1 and 2 are
    # in cluster 0 or 1.
    for cluster_id in range(config[salience_clustering.N_CLUSTERS_KEY]):
      top_tokens = [
          token_with_weight[0] for token_with_weight in result[
              salience_clustering.TOP_TOKEN_KEY]['input_embs_grad'][cluster_id]
      ]
      subset_cd = ['c', 'd']
      subset_ef = ['e', 'f']
      top_tokens_are_set_cd = subset_cd == top_tokens
      top_tokens_are_set_ef = subset_ef == top_tokens
      self.assertTrue(top_tokens_are_set_cd or top_tokens_are_set_ef)


if __name__ == '__main__':
  absltest.main()
