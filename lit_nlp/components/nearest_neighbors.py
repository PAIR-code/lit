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
"""Finds the k nearest neighbors to an input embedding."""


from typing import List, Optional, Sequence

import attr
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
import numpy as np
from scipy.spatial import distance


JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


@attr.s(auto_attribs=True, kw_only=True)
class NearestNeighborsConfig(object):
  """Config options for Nearest Neighbors component."""
  embedding_name: str = ''
  num_neighbors: Optional[int] = 10
  dataset_name: Optional[str] = ''


class NearestNeighbors(lit_components.Interpreter):
  """Computes nearest neighbors of an example embedding.

    Required Model Output:
      - Embeddings (`emb_layer`) to return the input embeddings
          for a layer
  """

  def run_with_metadata(
      self,
      indexed_inputs: Sequence[IndexedInput],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[List[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Finds the nearest neighbors of the example specified in the config.

    Args:
      indexed_inputs: the dataset example to find nearest neighbors for.
      model: the model being explained.
      dataset: the dataset which the current examples belong to.
      model_outputs: optional model outputs from calling model.predict(inputs).
      config: a config which should specify:
        {
          'num_neighbors': [the number of nearest neighbors to return]
          'dataset_name': [the name of the dataset (used for caching)]
          'embedding_name': [the name of the embedding field to use]
        }

    Returns:
      A JsonDict containing the a list of num_neighbors nearest neighbors,
      where each has the example id and distance from the main example.
    """
    config = NearestNeighborsConfig(**config)

    dataset_outputs = list(model.predict_with_metadata(
        dataset.indexed_examples, dataset_name=config.dataset_name))

    example_outputs = list(model.predict_with_metadata(
        indexed_inputs, dataset_name=config.dataset_name))
    # TODO(lit-dev): Add support for selecting nearest neighbors of a set.
    if len(example_outputs) != 1:
      raise ValueError('More than one selected example was passed in.')
    example_output = example_outputs[0]

    # <float32>[emb_size]
    dataset_embs = [output[config.embedding_name] for output in dataset_outputs]
    example_embs = [example_output[config.embedding_name]]
    distances = distance.cdist(example_embs, dataset_embs)[0]
    sorted_indices = np.argsort(distances)
    k = config.num_neighbors
    k_nearest_neighbors = [
        {'id': dataset.indexed_examples[original_index]['id'],
         'nn_distance': distances[original_index]
         } for original_index in sorted_indices[:k]]

    return [{'nearest_neighbors': k_nearest_neighbors}]
