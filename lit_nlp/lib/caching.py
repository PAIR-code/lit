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
"""Miscellaneous helper functions."""

import functools
import hashlib
import os
import pickle
import threading
from typing import Any, Callable, Iterable, Optional, Union

from absl import logging

from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import serialize

JsonDict = types.JsonDict
Input = types.Input
IndexedInput = types.IndexedInput

ProgressIndicator = Callable[[Iterable], Iterable]

# Compound keys: (dataset_name, example_id)
# None is used as a sentinel to skip the cache.
CacheKey = Union[tuple[str, str], None]

# The keys to the prediction locks are frozen sets of CacheKeys.
PredLockKey = frozenset[CacheKey]

# Special CacheKey to use when a model doesn't allow concurrent predictions.
PRED_LOCK_KEY_WHEN_NO_CONCURRENT_ACCESS: CacheKey = (
    "NO_CONCURRENT_PREDICTION", "")


def input_hash(example: Input) -> types.ExampleId:
  """Create stable hash of an input example."""
  json_str = serialize.to_json(
      example, simple=True, sort_keys=True).encode("utf-8")
  return types.ExampleId(hashlib.md5(json_str).hexdigest())


class PickleCacheLoader(object):
  """For saving and loading cache to a pickle file."""

  def __init__(self, name: str, cache_dir: str):
    self._cache_path = os.path.join(cache_dir, name + ".cache.pkl")

  def save(self, data: dict[CacheKey, Any]):
    with open(self._cache_path, "wb") as fd:
      pickle.dump(data, fd)

  def load(self) -> dict[CacheKey, Any]:
    """Load data from pickle file."""
    try:
      with open(self._cache_path, "rb") as fd:
        data = pickle.load(fd)
    except FileNotFoundError:
      logging.info("No cache to load at %s.", self._cache_path)
      data = {}
    except EOFError:
      logging.error(
          "Failed loading cache, possibly due to malformed cache data."
          "Please remove %s and try again.", self._cache_path)
      data = {}
    except IOError:
      logging.error("Failed loading cache at %s.", self._cache_path)
      data = {}
    return data  # pytype: disable=name-error  # py310-upgrade


class PredsCache(object):
  """Cache for model outputs."""

  def __init__(self, name: str, allow_concurrent_predictions: bool = True,
               cache_dir: Optional[str] = None):
    # TODO(lit-team): consider using a read/write lock, or setting timeouts if
    # contention becomes an issue.
    self._lock = threading.RLock()
    self._d: dict[CacheKey, Any] = dict()
    self._num_persisted = 0
    self._allow_concurrent_predictions = allow_concurrent_predictions

    self._cache_dir = cache_dir
    self._cache_loader = None
    if cache_dir:
      self._cache_loader = PickleCacheLoader(name, cache_dir)

    # A map of keys needing predictions to a lock for that model predict call.
    # Used for not duplicating concurrent prediction calls on the same inputs.
    self._pred_locks: dict[frozenset[CacheKey], threading.RLock] = dict()

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

  def info(self) -> str:
    """Print some info, for logging."""
    return str(len(self._d))

  def _construct_pred_lock_key(self, keys: list[CacheKey]) -> PredLockKey:
    # If this cache is set up to not allow concurrent predictions, then use the
    # same key to the predictions lock map regardless of example keys provided.
    fs = frozenset(keys) if self._allow_concurrent_predictions else frozenset(
        [PRED_LOCK_KEY_WHEN_NO_CONCURRENT_ACCESS])
    return fs

  def pred_lock_key(self, keys: list[CacheKey]) -> Optional[PredLockKey]:
    """Get the key for the predictions lock for the provided cache keys."""
    fs = self._construct_pred_lock_key(keys)

    # If the provided cache keys already have a lock, return the key.
    if fs in self._pred_locks:
      return fs
    # If there is a lock for a superset of the provided cache keys, return the
    # key to that lock.
    # This means that requests for subsets of data already being predicted will
    # wait for the larger set of predictions to complete. This can slow down
    # certain single-example requests but leads to more efficient use of the
    # model. We may have duplicate predict calls for an example to a model if
    # one example is part of separate but distinct predict calls with different
    # subsets of examples, but this is unlikely given how LIT predict requests
    # work.
    for key in self._pred_locks:
      if fs.issubset(key):
        return key
    # Otherwise, return None as there is no lock yet for the provided cache
    # keys.
    return None

  def get_pred_lock(self, keys: list[CacheKey]) -> threading.RLock:
    """Gets the lock for the provided cache keys, creating one if neccessary."""
    # If the lock already exists for the provided keys, return it.
    pl_key = self.pred_lock_key(keys)
    if pl_key:
      return self._pred_locks[pl_key]
    # If no such lock exists, create, store and return it.
    pred_lock = threading.RLock()
    self._pred_locks[self._construct_pred_lock_key(keys)] = pred_lock
    return pred_lock

  def delete_pred_lock(self, keys: list[CacheKey]) -> Optional[threading.RLock]:
    """Remove the lock from the map to clean up, returns it."""
    pl_key = self.pred_lock_key(keys)
    if pl_key is None:
      return None
    return self._pred_locks.pop(pl_key)

  def save_to_disk(self):
    """Save cache data to disk."""
    # No cache loader is created if no cache directory was provided, in which
    # case this is a no-op.
    if not self._cache_loader:
      return
    if self._num_persisted == len(self._d):
      logging.info("No need to re-save cache to %s", self._cache_dir)
      return
    logging.info("Saving cache (%d entries) to %s", len(self._d),
                 self._cache_dir)
    self._cache_loader.save(self._d)
    self._num_persisted = len(self._d)

  def load_from_disk(self):
    """Load cache data from disk."""
    # No cache loader is created if no cache directory was provided, in which
    # case this is a no-op.
    if not self._cache_loader:
      return
    self._d = self._cache_loader.load()
    self._num_persisted = len(self._d)
    logging.info("Loaded cache (%d entries) from %s", self._num_persisted,
                 self._cache_dir)


class CachingModelWrapper(lit_model.ModelWrapper):
  """Wrapper to add per-example caching to a LIT model."""

  def __init__(self,
               model: lit_model.Model,
               name: str,
               cache_dir: Optional[str] = None):
    """Wrap a model to add caching.

    Args:
      model: a LIT model
      name: name, used for logging and data files
      cache_dir: if given, will load/save data to disk
    """
    super().__init__(model)
    self._log_prefix = f"CachingModelWrapper '{name:s}'"
    self._cache = PredsCache(
        name, model.supports_concurrent_predictions, cache_dir)
    self.load_cache()

  def load_cache(self):
    self._cache.load_from_disk()

  def save_cache(self):
    self._cache.save_to_disk()

  def key_fn(self, d, group_name) -> CacheKey:
    if d["id"] == "":  # pylint: disable=g-explicit-bool-comparison
      logging.warning("Found empty example ID - using empty cache ID.")
      return None
    return (group_name, d["id"])

  ##
  # For internal use
  def fit_transform_with_metadata(self,
                                  indexed_inputs: list[JsonDict],
                                  dataset_name: str = ""):
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
    results = self._predict_with_metadata(*args, **kw)
    return results

  def _get_results_from_cache(self, input_keys: list[str]):
    with self._cache.lock:
      return [self._cache.get(input_key) for input_key in input_keys]

  def _predict_with_metadata(
      self,
      indexed_inputs: list[JsonDict],
      dataset_name: Optional[str] = None,
      progress_indicator: Optional[ProgressIndicator] = lambda x: x,
      **kw) -> list[JsonDict]:
    """As predict(), but inputs are IndexedInput."""
    # TODO(lit-dev): consider moving this to example level
    # (null keys skip cache), and removing this codepath.
    if dataset_name is None:
      logging.info("\n\nCache disabled for current call.\n\n")
      results = list(self.wrapped.predict_with_metadata(indexed_inputs))
      return results

    key_fn = functools.partial(self.key_fn, group_name=dataset_name)

    # Try to get results from the cache.
    input_keys = [key_fn(d) for d in indexed_inputs]
    if self._cache.pred_lock_key(input_keys):
      with self._cache.get_pred_lock(input_keys):
        results = self._get_results_from_cache(input_keys)
    else:
      results = self._get_results_from_cache(input_keys)

    # Make a single list of everything that wasn't found in the cache,
    # to actually run the model on these inputs.
    miss_idxs = [i for i, v in enumerate(results) if v is None]
    misses = [indexed_inputs[i] for i in miss_idxs]
    if misses:
      logging.info("%s: misses (dataset=%s): %s", self._log_prefix,
                   dataset_name, str([miss["id"] for miss in misses]))
      logging.info("%s: %d misses out of %d inputs", self._log_prefix,
                   len(miss_idxs), len(results))
    else:
      # If all results were already cached, return them.
      return results

    with self._cache.get_pred_lock(input_keys):
      model_preds = list(
          self.wrapped.predict_with_metadata(progress_indicator(misses)))
      logging.info("Received %d predictions from model", len(model_preds))

      if len(model_preds) != len(misses):
        raise ValueError(f"Received {len(model_preds)} predictions, which does "
                         f"not match {len(misses)}, the number of inputs.")

      # Merge results back into the output list.
      with self._cache.lock:
        for i, orig_idx in enumerate(miss_idxs):
          self._cache.put(model_preds[i], key_fn(indexed_inputs[orig_idx]))
          results[orig_idx] = model_preds[i]

      # Remove the prediction lock from the cache as the request is complete
      self._cache.delete_pred_lock(input_keys)

    return results
