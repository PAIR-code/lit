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
"""Uses nearest neighbor search for similar examples."""

from typing import List, Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import index

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput


class SimilaritySearcher(lit_components.Generator):
  """Searching by similarity."""

  def __init__(self, indexer: index.Indexer):
    self.index = indexer

  def _get_embedding(self, example: JsonDict, model: lit_model.Model,
                     dataset: lit_dataset.IndexedDataset, embedding_name: str,
                     dataset_name: str):
    """Calls the model on the example to get the embedding."""
    model_input = dataset.index_inputs([example])
    model_output = model.predict_with_metadata(
        model_input, dataset_name=dataset_name)
    embedding = list(model_output)[0][embedding_name]
    return embedding

  def _find_nn(self, model_name, dataset_name, embedding_name, embedding):
    """wrapper around the Index() class api."""
    similar_examples = self.index.find_nn(
        model_name, dataset_name, embedding_name, embedding, num_neighbors=25)
    return similar_examples

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.IndexedDataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Find similar examples for an example/model/dataset."""
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    embedding_name = config['Embedding Field']
    embedding = self._get_embedding(example, model, dataset, embedding_name,
                                    dataset_name)
    neighbors = self._find_nn(model_name, dataset_name, embedding_name,
                              embedding)
    return neighbors

  def config_spec(self) -> types.Spec:
    return {
        # Requires an embedding layer specified from a model.
        'Embedding Field': types.FieldMatcher(
            spec='output', types=['Embeddings'])
    }
