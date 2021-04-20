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
"""LIT model wrapper for pre-computed (offline) predictions."""
from typing import Optional, Iterable, Iterator, List

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import caching

JsonDict = lit_types.JsonDict


class StaticPredictions(lit_model.Model):
  """Implements lit.Model interface for a set of pre-computed predictions."""

  def key_fn(self, example: JsonDict) -> str:
    reduced_example = {k: example[k] for k in self.input_identifier_keys}
    return caching.input_hash(reduced_example)

  def description(self):
    return self._description

  @property
  def input_dataset(self):
    return self._all_inputs

  def __init__(self,
               inputs: lit_dataset.Dataset,
               preds: lit_dataset.Dataset,
               input_identifier_keys: Optional[List[str]] = None):
    """Build a static index.

    Args:
      inputs: a lit Dataset
      preds: a lit Dataset, parallel to inputs
      input_identifier_keys: (optional), list of keys to treat as identifiers
        for matching inputs. If None, will use all fields in inputs.spec()
    """
    self._all_inputs = inputs
    self._input_spec = inputs.spec()
    self._output_spec = preds.spec()
    self._description = preds.description()
    self.input_identifier_keys = input_identifier_keys or self._input_spec.keys(
    )
    # Filter to only the identifier keys
    self._input_spec = {
        k: self._input_spec[k] for k in self.input_identifier_keys
    }

    # Build the index for prediction lookups
    self._index = {
        self.key_fn(ex): pred
        for ex, pred in zip(inputs.examples, preds.examples)
    }

  def _predict_single(self, example: JsonDict):
    key = self.key_fn(example)
    if key not in self._index:
      raise KeyError(
          f'Example {key} not found in stored predictions: {str(example)}')
    return self._index[key]

  ##
  # LIT API implementation
  def input_spec(self):
    return self._input_spec

  def output_spec(self):
    return self._output_spec

  def predict_minibatch(self, inputs: List[JsonDict], **kw):
    return list(self.predict(inputs))

  def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterator[JsonDict]:
    """Predict on known inputs.

    Args:
      inputs: input examples
      **kw: unused

    Returns:
      predictions

    Raises:
      KeyError if input not recognized
    """
    # Implement predict() directly, since there's no need to batch.
    return map(self._predict_single, inputs)
