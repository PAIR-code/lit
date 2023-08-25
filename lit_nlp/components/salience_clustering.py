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
"""kmeans clustering of salience weights."""

from collections.abc import Sequence
from typing import Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
import numpy as np
from sklearn import cluster

IndexedInput = types.IndexedInput
JsonDict = types.JsonDict
Spec = types.Spec

# Result keys.
CLUSTER_ID_KEY = 'cluster_ids'
REPRESENTATION_KEY = 'representations'
TOP_TOKEN_KEY = 'top_tokens'

# Config items.
N_CLUSTERS_KEY = 'Number of clusters'
TOP_K_TOKENS_KEY = 'Number of top salient tokens to consider per data point'
N_TOKENS_TO_DISPLAY_KEY = 'Number of tokens to display per cluster'
SALIENCE_MAPPER_KEY = 'Salience Mapper'
SEED_KEY = 'Clustering seed'

# Config items not to need to be surfaced in the controls for this interpreter.
REUSE_CLUSTERING = 'reuse_clustering'


class SalienceClustering(lit_components.Interpreter):
  """Salience map clustering."""

  def __init__(self, salience_mappers: dict[str, lit_components.Interpreter]):
    self.salience_mappers = salience_mappers
    self.kmeans = {}
    self.vocab_lookup = {}
    self.vocab = []

  def _build_vocab(
      self,
      token_saliencies: list[JsonDict]) -> tuple[dict[str, int], list[str]]:
    """Build a vocabulary from the given token saliencies.

    This creates a mapping from word type to index in the vocabulary taken from
    all tokens in `token_saliencies`. The order of word types in the vocabulary
    depends on the order of the tokens in the input.

    Args:
      token_saliencies: List of mappings from salience field to TokenSaliency
        objects. This is the result of a post-hoc explanation method, such as
        gradient l2.

    Returns:
      1. Mapping from word type to its index in the vocabulary.
      2. Ordered list of word types.
    """
    vocab_lookup = {}

    for instance in token_saliencies:
      for token_salience in instance.values():
        for token in token_salience.tokens:
          vocab_lookup[token] = vocab_lookup.get(token, len(vocab_lookup))

    vocab = [''] * len(vocab_lookup)
    for token, idx in vocab_lookup.items():
      vocab[idx] = token
    return vocab_lookup, vocab

  def _convert_to_bow_vector(self, token_weights: JsonDict,
                             vocab_lookup: dict[str, int],
                             top_k: int) -> np.ndarray:
    """Converts the given variable length-vector into a fixed-length vector.

    This function creates a zero vector of the length of the vocabulary and
    fills the positions of the given tokens by their salience.

    Args:
      token_weights: Mapping from tokens to their salience weights.
      vocab_lookup: Mapping from word type to its index in the vocabulary.
      top_k: Number of tokens to be considered to represent each sentence.

    Returns:
      Vector of the size of the vocabulary that contains salience weights at
      the vocabulary indexes of tokens in `token_weights` and zeros elsewhere.
    """
    vocab_vector = np.zeros((len(vocab_lookup),))

    # Find k-th largest weight value
    sorted_token_weights = list(
        sorted(token_weights.items(), key=lambda x: abs(x[1]), reverse=True))
    index_end = min(top_k, len(sorted_token_weights))
    weight_threshold = abs(sorted_token_weights[index_end - 1][1])

    # Check if there exist ties
    for token_weight in sorted_token_weights[index_end:]:
      if abs(token_weight[1]) == weight_threshold:
        index_end += 1
      else:
        break

    # Consider only top-k (including ties)
    for token_weight in sorted_token_weights[:index_end]:
      if token_weight[0] in vocab_lookup:
        vocab_vector[vocab_lookup[token_weight[0]]] = token_weight[1]
    return vocab_vector

  def _compute_fixed_length_representation(
      self, token_saliencies: list[JsonDict],
      vocab_lookup: dict[str, int],
      top_k: int) -> list[dict[str, np.ndarray]]:
    """Compute a fixed-length representation from the variable-length salience.

    The representation is a simple vocabulary vector with salience weights as
    values. Every resulting vector is normalized to unit length.

    When a token occurs multiple times in the input we keep the value whose
    absolute value is largest.

    Args:
      token_saliencies: List of mappings from salience field to TokenSaliency
        objects. This is the result of a post-hoc explanation method, such as
        gradient l2.
      vocab_lookup: Mapping from word type to its index in the vocabulary.
      top_k: Number of tokens to be considered to represent each sentence. They
        will be determined based on their saliency scores.

    Returns:
      List of one mapping per example. Every element maps a salience field to
      its fixed-length representation.
    """
    representations = []
    for instance in token_saliencies:
      per_field_results = {}

      for salience_field, token_salience in instance.items():
        token_weights = {}

        for token, score in zip(token_salience.tokens, token_salience.salience):
          token_weights[token] = max([token_weights.get(token, score), score],
                                     key=abs)

        representation = self._convert_to_bow_vector(token_weights,
                                                     vocab_lookup, top_k)
        # Normalize to unit length.
        representation = np.asarray(representation) / np.linalg.norm(
            representation)
        per_field_results[salience_field] = representation
      representations.append(per_field_results)
    return representations

  def run(
      self,
      inputs: Sequence[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[JsonDict]:
    """Run this component, given a model and input(s).

    Note that when `config['REUSE_CLUSTERING'] == True` we reuse the previously
    computed kmeans objects and vocabulary. Tokens that do not exist in the
    vocabulary will be ignored.

    Args:
      inputs: Inputs to cluster.
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
        `TOP_TOKEN_KEY`: Top tokens are the tokens that have highest mean
          salience over all examples. They are sorted according to their
          salience.

    Raises:
      RuntimeError: Salience interpreter incompatible with model and dataset
      TypeError: config is not provided
      ValueError: config["Salience Mapper"] is not provided
    """

    if not config:
      raise TypeError('config must be provided')

    salience_key: Optional[str] = config.get(SALIENCE_MAPPER_KEY)
    if not salience_key:
      raise ValueError(f'config[{SALIENCE_MAPPER_KEY}] must be provided')

    if not (salience_interpreter := self.salience_mappers.get(salience_key)):
      raise ValueError(f'No interpreter registered for ${salience_key}.')

    if not salience_interpreter.is_compatible(model=model, dataset=dataset):
      raise RuntimeError(
          f'The {salience_key} interpreter is incompatible with this model and'
          ' dataset pair.'
      )

    # If no specific inputs provided, use the entire dataset.
    inputs_to_use = inputs or dataset.examples
    token_saliencies = salience_interpreter.run(
        inputs_to_use, model, dataset, model_outputs, config)

    if not token_saliencies:
      return None

    salience_fields = list(token_saliencies[0].keys())
    reuse_clustering = self.kmeans and config.get(REUSE_CLUSTERING, False)

    if not reuse_clustering:
      try:
        self.vocab = model.get_embedding_table()[0]
        self.vocab_lookup = {
            word_type: i for i, word_type in enumerate(self.vocab)
        }
      except NotImplementedError:
        self.vocab_lookup, self.vocab = self._build_vocab(token_saliencies)

    top_k = int(config.get(TOP_K_TOKENS_KEY,
                           self.config_spec()[TOP_K_TOKENS_KEY].default))

    representations = self._compute_fixed_length_representation(
        token_saliencies, self.vocab_lookup, top_k)

    cluster_ids = {}
    salience_field_to_representations = {}
    salience_field_to_top_tokens = {}

    for salience_field in salience_fields:
      weight_matrix = np.vstack([
          representation[salience_field] for representation in representations
      ])
      n_clusters = int(
          config.get(N_CLUSTERS_KEY,
                     self.config_spec()[N_CLUSTERS_KEY].default))
      seed = int(
          config.get(SEED_KEY,
                     self.config_spec()[SEED_KEY].default))
      if not reuse_clustering:
        self.kmeans[salience_field] = cluster.KMeans(
            n_clusters=n_clusters, random_state=seed)
        cluster_ids[salience_field] = self.kmeans[salience_field].fit_predict(
            weight_matrix).tolist()
      else:
        cluster_ids[salience_field] = self.kmeans[salience_field].predict(
            weight_matrix).tolist()

      salience_field_to_representations[salience_field] = weight_matrix
      salience_field_to_top_tokens[salience_field] = []

      for cluster_id in range(n_clusters):
        weight_matrix_of_cluster = weight_matrix[np.asarray(
            cluster_ids[salience_field]) == cluster_id]

        # If this is empty, we don't have any data points in the current
        # cluster. This may happen when `reuse_clustering` is true and we get
        # only a single example.
        if not weight_matrix_of_cluster.size:
          continue

        # <float32>[vocab size]
        mean_weight_matrix = weight_matrix_of_cluster.mean(axis=0)
        num_tokens_to_display = int(
            config.get(N_TOKENS_TO_DISPLAY_KEY,
                       self.config_spec()[N_TOKENS_TO_DISPLAY_KEY].default))
        top_indices = (
            np.abs(mean_weight_matrix).argsort()[::-1][:num_tokens_to_display])
        top_tokens = [
            (self.vocab[i], mean_weight_matrix[i])
            for i in top_indices if mean_weight_matrix[i] != 0.0
        ]
        salience_field_to_top_tokens[salience_field].append(top_tokens)

    return {
        CLUSTER_ID_KEY: cluster_ids,
        REPRESENTATION_KEY: salience_field_to_representations,
        TOP_TOKEN_KEY: salience_field_to_top_tokens,
    }

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    return any(
        interp.is_compatible(model=model, dataset=dataset)
        for interp in self.salience_mappers.values())

  def config_spec(self) -> types.Spec:
    return {
        SALIENCE_MAPPER_KEY:
            types.CategoryLabel(
                required=True, vocab=list(self.salience_mappers.keys())),
        N_CLUSTERS_KEY:
            types.Scalar(min_val=2, max_val=100, default=10, step=1),
        TOP_K_TOKENS_KEY:
            types.Scalar(min_val=1, max_val=20, default=5, step=1),
        N_TOKENS_TO_DISPLAY_KEY:
            types.Scalar(min_val=5, max_val=50, default=10, step=1),
        SEED_KEY:
            types.TextSegment(default='0'),
    }

  def meta_spec(self) -> types.Spec:
    return {
        CLUSTER_ID_KEY: types.CategoryLabel(),
        REPRESENTATION_KEY: types.Embeddings(),
        TOP_TOKEN_KEY: types.TopTokens(),
    }
