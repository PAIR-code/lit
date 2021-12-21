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
"""kmeans clustering of salience weights."""

from typing import Dict, List, Optional, Sequence, Text

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np
from sklearn import cluster

IndexedInput = types.IndexedInput
JsonDict = types.JsonDict
Spec = types.Spec

CLUSTER_ID_KEY = 'cluster_ids'
REPRESENTATION_KEY = 'representations'

N_CLUSTERS_KEY = 'Number of Clusters'
SALIENCE_MAPPER_KEY = 'Salience Mapper'


class SalienceClustering(lit_components.Interpreter):
  """Salience map clustering."""

  def __init__(self, salience_mappers: Dict[str, lit_components.Interpreter]):
    self.salience_mappers = salience_mappers
    self.kmeans = {}

  def find_fields(self, output_spec: Spec) -> List[Text]:
    # Find TokenGradients fields
    grad_fields = utils.find_spec_keys(output_spec, types.TokenGradients)

    # Check that these are aligned to Tokens fields
    for f in grad_fields:
      tokens_field = output_spec[f].align  # pytype: disable=attribute-error
      assert tokens_field in output_spec
      assert isinstance(output_spec[tokens_field], types.Tokens)
    return grad_fields

  def _build_vocab(self, token_saliencies: List[JsonDict]) -> Dict[str, int]:
    """Build a vocabulary from the given token saliencies.

    This creates a mapping from word type to index in the vocabulary taken from
    all tokens in `token_saliencies`. The order of word types in the vocabulary
    depends on the order of the tokens in the input.

    Args:
      token_saliencies: List of mappings from gradient field to TokenSaliency
        objects. This is the result of a post-hoc explanation method, such as
        gradient l2.

    Returns:
      Mapping from word type to its index in the vocabulary.
    """
    vocab = {}

    for instance in token_saliencies:
      for token_salience in instance.values():
        for token in token_salience.tokens:
          vocab[token] = vocab.get(token, len(vocab))

    return vocab

  def _convert_to_bow_vector(self, token_weights: JsonDict,
                             vocab: Dict[str, int]) -> np.ndarray:
    """Converts the given variable length-vector into a fixed-length vector.

    This function creates a zero vector of the length of the vocabulary and
    fills the positions of the given tokens by their salience.

    Args:
      token_weights: Mapping from tokens to their salience weights.
      vocab: Mapping from word type to its index in the vocabulary.

    Returns:
      Vector of the size of the vocabulary that contains salience weights at
      the vocabulary indexes of tokens in `token_weights` and zeros elsewhere.
    """
    vocab_vector = np.zeros((len(vocab),))

    for token, token_weight in token_weights.items():
      vocab_vector[vocab[token]] = token_weight
    return vocab_vector

  def _compute_fixed_length_representation(
      self, token_saliencies: List[JsonDict],
      vocab: Dict[str, int]) -> List[Dict[str, np.ndarray]]:
    """Compute a fixed-length representation from the variable-length salience.

    The representation is a simple vocabulary vector with salience weights as
    values. Every resulting vector is normalized to unit length.

    When a token occurs multiple times in the input we keep the value whose
    absolute value is largest.

    Args:
      token_saliencies: List of mappings from gradient field to TokenSaliency
        objects. This is the result of a post-hoc explanation method, such as
        gradient l2.
      vocab: Mapping from word type to its index in the vocabulary.

    Returns:
      List of one mapping per example. Every element maps a gradient field to
      its fixed-length representation.
    """
    representations = []
    for instance in token_saliencies:
      per_field_results = {}

      for grad_field, token_salience in instance.items():
        token_weights = {}

        for token, score in zip(token_salience.tokens, token_salience.salience):
          token_weights[token] = max([token_weights.get(token, score), score],
                                     key=abs)

        representation = self._convert_to_bow_vector(token_weights, vocab)
        # Normalize to unit length.
        representation = np.asarray(representation) / np.linalg.norm(
            representation)
        per_field_results[grad_field] = representation
      representations.append(per_field_results)
    return representations

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    """Run this component, given a model and input(s)."""
    raise NotImplementedError(
        'Not implemented. Call run_with_metadata() directly.')

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[List[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[JsonDict]:
    """Run this component, given a model and input(s).

    Args:
      indexed_inputs: Inputs to cluster.
      model: Model that provides salience maps.
      dataset: Dataset to compute salience maps for.
      model_outputs: Precomputed model outputs.
      config: Config for clustering and salience computation

    Returns:
      Dict with 2 keys:
        `CLUSTER_ID_KEY`: Contains the cluster assignments. One cluster id per
          dataset example.
        `REPRESENTATION_KEY`: Contains the representations of all examples in
          the dataset that were used in the clustering.
    """
    config = config or {}
    # Find gradient fields to interpret
    grad_fields = self.find_fields(model.output_spec())
    token_saliencies = self.salience_mappers[
        config['salience_mapper']].run_with_metadata(indexed_inputs, model,
                                                     dataset, model_outputs,
                                                     config)

    if not token_saliencies:
      return None

    vocab = self._build_vocab(token_saliencies)
    representations = self._compute_fixed_length_representation(
        token_saliencies, vocab)

    cluster_ids = {}
    grad_field_to_representations = {}

    for grad_field in grad_fields:
      weight_matrix = np.vstack(
          representation[grad_field] for representation in representations)
      self.kmeans[grad_field] = cluster.KMeans(
          n_clusters=config.get(N_CLUSTERS_KEY,
                                self.config_spec()[N_CLUSTERS_KEY].default))
      cluster_ids[grad_field] = self.kmeans[grad_field].fit_predict(
          weight_matrix).tolist()
      grad_field_to_representations[grad_field] = weight_matrix

    return {
        CLUSTER_ID_KEY: cluster_ids,
        REPRESENTATION_KEY: grad_field_to_representations
    }

  def is_compatible(self, model: lit_model.Model):
    compatible_fields = self.find_fields(model.output_spec())
    return len(compatible_fields)

  def config_spec(self) -> types.Spec:
    return {
        SALIENCE_MAPPER_KEY:
            types.CategoryLabel(
                required=True, vocab=list(self.salience_mappers.keys())),
        N_CLUSTERS_KEY:
            types.Scalar(min_val=2, max_val=100, default=2, step=1),
    }

  def meta_spec(self) -> types.Spec:
    return {
        CLUSTER_ID_KEY: types.CategoryLabel(),
        REPRESENTATION_KEY: types.Embeddings(),
    }
