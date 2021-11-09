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
"""Miscellaneous helper functions."""

import functools
import hashlib
import os
import pickle
import threading
from typing import Text, Optional, Union, Any, List, Tuple

from absl import logging

from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import serialize

JsonDict = types.JsonDict
Input = types.Input
IndexedInput = types.IndexedInput

# Compound keys: (dataset_name, example_id)
# None is used as a sentinel to skip the cache.
CacheKey = Union[Tuple[Text, Text], None]


def input_hash(example: JsonDict) -> Text:
  """Create stable hash of an input example."""
  json_str = serialize.to_json(
      example, simple=True, sort_keys=True).encode("utf-8")
  return hashlib.md5(json_str).hexdigest()


class PredsCache(object):
  """Cache for model outputs."""

  def __init__(self):
    # TODO(lit-team): consider using a read/write lock, or setting timeouts if
    # contention becomes an issue.
    self._lock = threading.RLock()
    # TODO(lit-dev): consider using an OrderedDict to implement a LRU cache of
    # bounded size.
    self._d = dict()

  @property
  def lock(self):
    return self._lock

  def put(self, data, key: CacheKey):
    if key is None:
      logging.info("Ignoring put(data, None) due to sentinel values in key.")
      return
    self._d[key] = data

  def get(self, key: CacheKey) -> Optional[Any]:
    if key is None:
      logging.info("Ignoring get(None) due to sentinel values in key.")
      return None
    return self._d.get(key, None)

  def info(self) -> Text:
    """Print some info, for logging."""
    return str(len(self._d))

  ##
  # For development use
  def save_to_disk(self, path):
    """Save cache data to disk."""
    logging.info("Saving cache (%d entries) to %s", len(self._d), path)
    with open(path, "wb") as fd:
      pickle.dump(self._d, fd)

  def load_from_disk(self, path):
    """Load cache data from disk."""
    try:
      with open(path, "rb") as fd:
        data = pickle.load(fd)
      self._d = data
      logging.info("Loaded cache (%d entries) from %s", len(self._d), path)
    except EOFError:
      logging.error(
          "Failed loading cache, possibly due to malformed cache data."
          "Please remove %s and try again.", path)
      exit(1)


class CachingModelWrapper(lit_model.ModelWrapper):
  """Wrapper to add per-example caching to a LIT model."""

  def __init__(self,
               model: lit_model.Model,
               name: Text,
               cache_dir: Optional[Text] = None):
    """Wrap a model to add caching.

    Args:
      model: a LIT model
      name: name, used for logging and data files
      cache_dir: if given, will load/save data to disk
    """
    super().__init__(model)
    self._log_prefix = f"CachingModelWrapper '{name:s}'"
    self._cache = PredsCache()
    self._cache_path = None
    if cache_dir:
      self._cache_path = os.path.join(cache_dir, name + ".cache.pkl")
    self.load_cache()

  def load_cache(self):
    if not self._cache_path:
      logging.info("%s: no cache path specified, not loading.",
                   self._log_prefix)
      return

    if not os.path.exists(self._cache_path):
      logging.info("%s: cache file %s does not exist, not loading.",
                   self._log_prefix, self._cache_path)
      return

    logging.info("%s: loading from %s", self._log_prefix, self._cache_path)
    self._cache.load_from_disk(self._cache_path)

  def save_cache(self):
    if not self._cache_path:
      logging.info("%s: no cache path specified, not saving.", self._log_prefix)
      return

    logging.info("%s: saving to %s", self._log_prefix, self._cache_path)
    self._cache.save_to_disk(self._cache_path)

  def key_fn(self, d, group_name) -> CacheKey:
    if d["id"] == "":  # pylint: disable=g-explicit-bool-comparison
      logging.warning("Found empty example ID - using empty cache ID.")
      return None
    return (group_name, d["id"])

  ##
  # For internal use
  def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict],
                                  dataset_name: Text):
    """For use with UMAP and other preprocessing transforms."""
    outputs = list(self.wrapped.fit_transform_with_metadata(indexed_inputs))
    key_fn = functools.partial(self.key_fn, group_name=dataset_name)
    with self._cache.lock:
      for i, output in enumerate(outputs):
        self._cache.put(output, key_fn(indexed_inputs[i]))
    return outputs

  def predict_minibatch(self, *args, **kw):
    logging.warning(
        "CachingModelWrapper.predict_minibatch() bypasses the cache - "
        "if this is not intended, use predict_with_metadata() instead "
        "to access cache via example IDs.")
    return self.wrapped.predict_minibatch(*args, **kw)

  def predict(self, *args, **kw):
    logging.warning(
        "CachingModelWrapper.predict() bypasses the cache - "
        "if this is not intended, use predict_with_metadata() instead "
        "to access cache via example IDs.")
    return self.wrapped.predict(*args, **kw)

  def predict_with_metadata(self, *args, **kw):
    """As predict(), but inputs are IndexedInput."""
    # Lock for the entire request, to avoid running the model more than once
    # on the same inputs. This shouldn't cause much of a performance hit, since
    # models are generally compute-bound anyway.
    with self._cache.lock:
      results = self._predict_with_metadata(*args, **kw)
    return results

  def _predict_with_metadata(self,
                             indexed_inputs: List[JsonDict],
                             dataset_name: Optional[Text] = None,
                             **kw) -> List[JsonDict]:
    """As predict(), but inputs are IndexedInput."""
    # TODO(lit-dev): consider moving this to example level
    # (null keys skip cache), and removing this codepath.
    if dataset_name is None:
      logging.info("\n\nCache disabled for current call.\n\n")
      results = list(self.wrapped.predict_with_metadata(indexed_inputs))
      return results

    key_fn = functools.partial(self.key_fn, group_name=dataset_name)

    # Try to get results from the cache.
    results = [self._cache.get(key_fn(d)) for d in indexed_inputs]
    miss_idxs = [i for i, v in enumerate(results) if v is None]
    logging.info("%s: misses (dataset=%s): %s", self._log_prefix, dataset_name,
                 str([indexed_inputs[i]["id"] for i in miss_idxs]))
    logging.info("%s: %d misses out of %d inputs", self._log_prefix,
                 len(miss_idxs), len(results))

    # Make a single list of everything that wasn't found in the cache,
    # and actually run the model on these inputs.
    model_inputs = [indexed_inputs[i] for i in miss_idxs]
    logging.info("Prepared %d inputs for model", len(model_inputs))
    model_preds = list(self.wrapped.predict_with_metadata(model_inputs))
    logging.info("Received %d predictions from model", len(model_preds))
    assert len(model_preds) == len(
        model_inputs
    ), f"Received {len(model_preds)} predictions, which does not match {len(model_inputs)}, the number of inputs."

    # Merge results back into the output list.
    for i, orig_idx in enumerate(miss_idxs):
      self._cache.put(model_preds[i], key_fn(indexed_inputs[orig_idx]))
      results[orig_idx] = model_preds[i]

    return results
