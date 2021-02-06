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
"""Indexer class for fast nearest neighbor lookups."""

import collections
import os
# TODO(b/151080311): don't use pickle for this.
import pickle
from typing import Optional, Text, List, Mapping

from absl import logging
import annoy

from lit_nlp.api import dataset as lit_data
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils


class Indexer(object):
  """Class to build and search annoy indices.

  TODO(b/162421415): Split the managing of indices and the index logic.

  Upon instantiation of this class, it will first create empty indices for
  every model/embedding/dataset triple. It will then search for existing indices
  in the data directory and fill those. If the flag `initialize_new_indices` is
  set, it will fill the remaining indices by iterating over the data.
  During the saving process, both the index and a mapping from index-row to
  example are saved. These are used during the nearest neighbor lookup to return
  the closest example.

  Attributes:
     models: specification akin to the LIT server.
     datasets: specification akin to the LIT server.
     data_dir: path for (de-)serialization of indices.
     initialize_new_indices: whether to build new indices or simply load
       existing ones.
  """

  def __init__(
      self,
      models: Mapping[Text, lit_model.Model],
      datasets: Mapping[Text, lit_data.IndexedDataset],
      data_dir: Optional[Text],
      initialize_new_indices: Optional[bool] = False,
  ):
    self.datasets = datasets
    self._indices = collections.OrderedDict()
    self._example_lookup = collections.defaultdict(dict)
    # Indicator whether to build new indices. If False, only load existing ones.
    self._initialize_new_indices = initialize_new_indices
    # Ensure directory to save indices exists.
    if not os.path.isdir(data_dir):
      os.mkdir(data_dir)
    self._data_dir = data_dir
    self._models = models

    # Create/Load indices.
    for model_name, model_info in self._models.items():
      compatible_datasets = [
          dname for dname, ds in self.datasets.items()
          if model_info.spec().is_compatible_with_dataset(ds.spec())
      ]
      for dataset in compatible_datasets:
        self._create_empty_indices(model_name, dataset)
        self._fill_indices(model_name, dataset)

    # Update cache with all indices.
    self._save_lookups()

  def _get_index_key(self, model_name, dataset_name, embedding_name):
    """Returns the key of an index, added to avoid collisions."""
    index_key = f"{dataset_name}:{model_name}:{embedding_name}"
    return index_key

  def _get_lookup_key(self, model_name, dataset_name):
    """Returns the key of a text lookup table."""
    lookup_key = f"{dataset_name}:{model_name}"
    return lookup_key

  def _get_index_path(self, index_key):
    """Get the file path for an index."""
    file_path = os.path.join(self._data_dir, f"{index_key}.ann")
    return file_path

  def _get_lookup_path(self, lookup_key):
    """Get the file path for the lookup index."""
    file_path = os.path.join(self._data_dir, lookup_key + ".pkl")
    return file_path

  def _create_empty_indices(self, model_name, dataset_name):
    """Create the empty indices for a model and dataset."""
    model = self._models[model_name]
    examples = self.datasets[dataset_name].indexed_examples
    model_embeddings_names = utils.find_spec_keys(model.output_spec(),
                                                  lit_types.Embeddings)
    if not model_embeddings_names:
      return

    # To first create an index, we need to know the shapes - peek at first ex.
    peeked_example = list(model.predict([examples[0]["data"]]))[0]
    for emb_name in model_embeddings_names:
      index_key = self._get_index_key(model_name, dataset_name, emb_name)
      emb_dimension = len(peeked_example[emb_name])
      assert self._indices.get(index_key) is None, "Index already exists."
      self._indices[index_key] = annoy.AnnoyIndex(emb_dimension, "euclidean")

  def _load_lookup(self, lookup_key):
    """Loads a lookup table from index to data example."""
    lookup_path = self._get_lookup_path(lookup_key)
    if not os.path.exists(lookup_path):
      return {}
    with open(lookup_path, "rb") as f:
      return pickle.load(f)

  def _fill_indices(self, model_name, dataset_name):
    """Create all indices for a single model."""
    model = self._models.get(model_name)
    assert model is not None, "Invalid model name."
    examples = self.datasets[dataset_name].indexed_examples
    model_embeddings_names = utils.find_spec_keys(model.output_spec(),
                                                  lit_types.Embeddings)
    lookup_key = self._get_lookup_key(model_name, dataset_name)

    # If the model has no embeddings to extract, skip the following.
    if not model_embeddings_names:
      return

    # Load from file if it exists.
    for emb_name in model_embeddings_names:
      # Initialize the index object in self._indices with serialized index.
      self._init_index_from_file(model_name, dataset_name, emb_name)
    # Load example lookup dictionary from file.
    self._example_lookup[lookup_key] = self._load_lookup(lookup_key)

    # Identify which indices need to be initialized.
    embeddings_to_index = [
        emb_name for emb_name in model_embeddings_names
        if not self._is_index_initialized(model_name, dataset_name, emb_name)
    ]
    # Early exit if all embeddings are now initialized.
    if not embeddings_to_index:
      return

    # Cold start: Get embeddings for non-initialized settings.
    if self._initialize_new_indices:
      for res_ix, (result, example) in enumerate(
          zip(model.predict_with_metadata(examples), examples)):
        for emb_name in embeddings_to_index:
          index_key = self._get_index_key(model_name, dataset_name, emb_name)
          # Initialize saving in the first iteration.
          if res_ix == 0:
            file_path = self._get_index_path(index_key)
            self._indices[index_key].on_disk_build(file_path)
          index = self._indices.get(index_key)
          assert index is not None, "Index needs to be created first."
          # Each item has an incrementing ID res_ix.
          self._indices[index_key].add_item(res_ix, result[emb_name])
        # Add item to lookup table.
        self._example_lookup[lookup_key][res_ix] = example["data"]

      # Create the trees from the indices - using 10 as recommended by doc.
      for emb_name in embeddings_to_index:
        index_key = self._get_index_key(model_name, dataset_name, emb_name)
        logging.info("Creating new index: %s", index_key)
        self._indices[index_key].build(10)
        index_size = self._indices[index_key].get_n_items()
        logging.info("Created new index with %s items.", index_size)

  def _is_index_initialized(self, model_name, dataset_name, emb_name):
    """Checks if an index is already initialized (num trees > 0)."""
    index_key = self._get_index_key(model_name, dataset_name, emb_name)
    return self._indices[index_key].get_n_trees() > 0

  def _init_index_from_file(self, model_name, dataset_name, emb_name):
    """If possible, load indices from file for a model/data combination."""
    index_key = self._get_index_key(model_name, dataset_name, emb_name)
    index_path = self._get_index_path(index_key)
    if os.path.exists(index_path):
      logging.info("Loading from cache: %s", index_path)
      self._indices[index_key].load(index_path)
      index_size = self._indices[index_key].get_n_items()
      logging.info("Loaded index with %s items.", index_size)

  def _save_lookups(self):
    """Iterate over indices and lookup tables and save them."""
    # Save the lookup tables.
    for lookup_key, lookup_table in self._example_lookup.items():
      file_path = self._get_lookup_path(lookup_key)
      with open(file_path, "wb") as f:
        pickle.dump(lookup_table, f, pickle.HIGHEST_PROTOCOL)

  def find_nn(self,
              model_name: Text,
              dataset_name: Text,
              embedding_name: Text,
              embedding: List[float],
              num_neighbors: Optional[int] = 25):
    """Find the nearest neighbor in index for an embedding.

    This function implements the search API for this class.
    The model/dataset/embedding combination maps to a unique index. Within, we
    are looking up the num_neighbors nearest neighbors of the embedding arg.

    Args:
      model_name: The identifier of the model.
      dataset_name: The identifier of the dataset.
      embedding_name: The identifier of the embedding within the model spec.
      embedding: The embedding we aim to look up.
      num_neighbors: How many neighbors we should return.

    Returns:
      neighbors: A list[dict] with nearest neighbor examples.
    """
    index_key = self._get_index_key(model_name, dataset_name, embedding_name)
    index = self._indices.get(index_key)
    assert index is not None, f"No index found for {index_key}."

    # Query for the neighbors.
    neighbor_indices, distances = index.get_nns_by_vector(
        vector=embedding, n=num_neighbors, include_distances=True)

    # Convert neighbors to texts.
    lookup_key = self._get_lookup_key(model_name, dataset_name)
    neighbor_examples = [
        self._example_lookup[lookup_key][ix] for ix in neighbor_indices
    ]

    # TODO(lit-dev): make the distance metadata that can be returned.
    del distances

    return neighbor_examples
