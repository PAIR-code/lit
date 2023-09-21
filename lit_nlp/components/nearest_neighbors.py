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
"""Finds the k nearest neighbors to an input embedding."""

from collections.abc import Sequence
import dataclasses
from typing import Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np
from scipy.spatial import distance


JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


@dataclasses.dataclass
class NearestNeighborsConfig(object):
  """Config options for Nearest Neighbors component."""
  embedding_name: str = ''
  num_neighbors: Optional[int] = 10
  use_input: Optional[bool] = False


_NN_CONFIG_FIELDS = [
    field.name for field in dataclasses.fields(NearestNeighborsConfig)]


class NearestNeighbors(lit_components.Interpreter):
  """Computes nearest neighbors of an example embedding.

  Required Model Output:
    - Embeddings (`emb_layer`) to return the input embeddings
        for a layer
  """

  def is_compatible(
      self, model: lit_model.Model, dataset: lit_dataset.Dataset
  ) -> bool:
    dataset_embs = utils.spec_contains(dataset.spec(), types.Embeddings)
    model_out_embs = utils.spec_contains(model.output_spec(), types.Embeddings)
    return dataset_embs or model_out_embs

  def run(
      self,
      inputs: Sequence[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Finds the nearest neighbors of the example specified in the config.

    Args:
      inputs: the dataset example to find nearest neighbors for.
      model: the model being explained.
      dataset: the dataset which the current examples belong to.
      model_outputs: optional model outputs from calling model.predict(inputs).
      config: a config which should specify:
        {
          'num_neighbors': [the number of nearest neighbors to return]
          'embedding_name': [the name of the embedding field to use]
          'use_input': [Optional boolean if the embedding comes from input data]
        }

    Raises:
      KeyError: Cannot find the embedding field in the relevant spec
      TypeError: `config` argument not provided
      ValueError: indexed_inputs requires exactly one input

    Returns:
      A JsonDict containing the a list of num_neighbors nearest neighbors,
      where each has the example id and distance from the main example.
    """
    if not config:
      raise TypeError('config must be provided')

    if not (isinstance(dataset, lit_dataset.IndexedDataset)):
      raise TypeError('Nearest neighbors requires an IndexedDataset to track '
                      'uniqueness by ID.')

    nnconf = NearestNeighborsConfig(**{
        k: v for k, v in config.items() if k in _NN_CONFIG_FIELDS
    })

    # TODO(lit-dev): Add support for selecting nearest neighbors of a set.
    if len(inputs) != 1:
      raise ValueError('indexed_inputs must contain exactly 1 example, found '
                       f'{len(inputs)}.')

    if nnconf.use_input:
      if not dataset.spec().get(nnconf.embedding_name):
        raise KeyError('Could not find embeddings field, '
                       f'{nnconf.embedding_name} in dataset spec')
      # If using input values, then treat inputs as outputs instead of running
      # the model.
      dataset_outputs = dataset.examples
      example_outputs = inputs
    else:
      if not model.output_spec().get(nnconf.embedding_name):
        raise KeyError('Could not find embeddings field, '
                       f'{nnconf.embedding_name} in model output spec')
      dataset_outputs = list(model.predict(dataset.examples))
      example_outputs = list(model.predict(inputs))

    example_output = example_outputs[0]

    # <float32>[emb_size]
    dataset_embs = [output[nnconf.embedding_name] for output in dataset_outputs]
    example_embs = [example_output[nnconf.embedding_name]]
    distances = distance.cdist(example_embs, dataset_embs)[0]
    sorted_indices = np.argsort(distances)
    k = nnconf.num_neighbors
    k_nearest_neighbors = [
        {'id': dataset.examples[original_index]['_id'],
         'nn_distance': distances[original_index]
         } for original_index in sorted_indices[:k]]

    return [{'nearest_neighbors': k_nearest_neighbors}]
