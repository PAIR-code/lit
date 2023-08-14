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

from collections.abc import Callable, Iterable
import os
import pickle
import threading
from typing import Any, Optional, Union

from absl import logging
from lit_nlp.api import dataset as lit_dataset
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

input_hash = lit_dataset.input_hash


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
    self._pred_locks: dict[PredLockKey, threading.RLock] = dict()

  @property
  def lock(self):
    return self._lock

  def put(self, data, key: CacheKey):
    if key is not None:
      self._d[key] = data

  def get(self, key: CacheKey) -> Optional[Any]:
    return self._d.get(key) if key is not None else None

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
    cache_loader = self._cache_loader
    if not cache_loader:
      return
    if self._num_persisted == len(self._d):
      logging.info("No need to re-save cache to %s", self._cache_dir)
      return
    logging.info(
        "Saving cache (%d entries) to %s", len(self._d), self._cache_dir
    )
    cache_loader.save(self._d)
    self._num_persisted = len(self._d)

  def load_from_disk(self):
    """Load cache data from disk."""
    # No cache loader is created if no cache directory was provided, in which
    # case this is a no-op.
    cache_loader = self._cache_loader
    if not cache_loader:
      return
    self._d = cache_loader.load()
    self._num_persisted = len(self._d)
    logging.info(
        "Loaded cache (%d entries) from %s",
        self._num_persisted,
        self._cache_dir,
    )


class CachingModelWrapper(lit_model.ModelWrapper):
  """Wrapper to add per-example caching to a LIT model."""

  def __init__(
      self,
      model: lit_model.Model,
      name: str,
      cache_dir: Optional[str] = None,
      strict_id_validation: bool = False,
      id_hash_fn: Optional[lit_dataset.IdFnType] = None,
  ):
    """Wrap a model to add caching.

    Args:
      model: a LIT model
      name: name, used for logging and data files
      cache_dir: if given, will load/save data to disk
      strict_id_validation: if true, will re-compute hashes using id_hash_fn and
        verify that they match the provided IDs. See b/293984290.
      id_hash_fn: function of example --> string id, used by
        strict_id_validation mode.
    """
    if strict_id_validation and id_hash_fn is None:
      raise ValueError(
          "Must provide id_hash_fn to use strict_id_validation mode."
      )

    super().__init__(model)
    self._name = name
    self._log_prefix = f"CachingModelWrapper '{name:s}'"
    self._strict_id_validation = strict_id_validation
    self._id_hash_fn = id_hash_fn
    self._cache = PredsCache(
        name, model.supports_concurrent_predictions, cache_dir
    )
    self.load_cache()

  def load_cache(self):
    self._cache.load_from_disk()

  def save_cache(self):
    self._cache.save_to_disk()

  def key_fn(self, d) -> CacheKey:
    return (self._name, d_id) if (d_id := d.get("_id")) else None

  def _validate_ids(self, inputs: Iterable[JsonDict]):
    for ex in inputs:
      if not (given_id := ex.get("_id")):
        continue
      if (computed_id := self._id_hash_fn(types.Input(ex))) != given_id:
        raise ValueError(
            f"Given id '{given_id}' does not match computed id '{computed_id}'"
            f" for example {str(ex)}."
        )

  ##
  # For internal use
  def fit_transform(self, inputs: Iterable[JsonDict]):
    """Cache projections from ProjectorModel dimensionality reducers."""
    wrapped = self.wrapped
    if not isinstance(wrapped, lit_model.ProjectorModel):
      raise TypeError(
          "Attempted to call fit_transform() on a non-ProjectorModel."
      )

    inputs_as_list = list(inputs)
    cache_keys = [self.key_fn(d) for d in inputs_as_list]
    if (none_keys := [k for k in cache_keys if k is None]):
      logging.warning(
          "Attmepting to cache %d (of %d) where the cache key is None "
          "- this can be from a missing or empty example id. These"
          " will be recomputed on subsequent attempts.",
          len(none_keys),
          len(cache_keys),
      )
    outputs = list(wrapped.fit_transform(inputs_as_list))
    with self._cache.lock:
      for cache_key, output in zip(cache_keys, outputs, strict=True):
        self._cache.put(output, cache_key)
    return outputs

  def predict(self,
              inputs: Iterable[JsonDict],
              progress_indicator: Optional[ProgressIndicator] = lambda x: x,
              **kw) -> list[JsonDict]:
    inputs_as_list = list(inputs)

    if self._strict_id_validation:
      self._validate_ids(inputs_as_list)

    # Try to get results from the cache.
    input_keys = [self.key_fn(d) for d in inputs_as_list]
    if (none_keys := [k for k in input_keys if k is None]):
      logging.warning(
          "Attmepting to retrieve %d (of %d) predictions from the cache where"
          " the cache key is None - this can be from a missing or empty example"
          " id. These will call model.predict() on this and subsequent calls.",
          len(none_keys),
          len(input_keys),
      )
    if self._cache.pred_lock_key(input_keys):
      with self._cache.get_pred_lock(input_keys):
        cached_results = self._get_results_from_cache(input_keys)
    else:
      cached_results = self._get_results_from_cache(input_keys)

    # Make a single list of everything that wasn't found in the cache,
    # to actually run the model on these inputs.
    miss_idxs = [i for i, v in enumerate(cached_results) if v is None]
    misses = [inputs_as_list[i] for i in miss_idxs]
    if misses:
      logging.info("%s: %d misses out of %d inputs", self._log_prefix,
                   len(miss_idxs), len(cached_results))
    else:
      # If all results were already cached, return them.
      return cached_results

    with self._cache.get_pred_lock(input_keys):
      model_preds = list(self.wrapped.predict(progress_indicator(misses)))
      logging.info("Received %d predictions from model", len(model_preds))

      if len(model_preds) != len(misses):
        raise ValueError(f"Received {len(model_preds)} predictions, which does "
                         f"not match {len(misses)}, the number of inputs.")

      # Merge results back into the output list.
      with self._cache.lock:
        for i, orig_idx in enumerate(miss_idxs):
          self._cache.put(model_preds[i], self.key_fn(inputs_as_list[orig_idx]))
          cached_results[orig_idx] = model_preds[i]

      # Remove the prediction lock from the cache as the request is complete
      self._cache.delete_pred_lock(input_keys)

    return cached_results

  def _get_results_from_cache(self, input_keys: list[CacheKey]):
    with self._cache.lock:
      return [self._cache.get(input_key) for input_key in input_keys]
