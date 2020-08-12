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

import copy

from typing import Dict, List, TypeVar, Callable, Any

K = TypeVar('K')
V = TypeVar('V')


def coerce_bool(value) -> bool:
  if isinstance(value, (bool, int, float, list, dict)):
    return bool(value)
  elif value is None:
    return False
  elif str(value).lower() in ['', '0', 'false']:
    return False
  else:
    return True


def find_keys(d: Dict[K, V], predicate: Callable[[V], bool]) -> List[K]:
  """Find keys where values match predicate."""
  return [k for k, v in d.items() if predicate(v)]


def find_spec_keys(d: Dict[K, Any], types) -> List[K]:
  """Find keys where values match one or more types."""
  return find_keys(d, lambda v: isinstance(v, types))


def filter_by_keys(d: Dict[K, V], predicate: Callable[[K], bool]) -> Dict[K, V]:
  """Filter to keys matching predicate."""
  return {k: v for k, v in d.items() if predicate(k)}


def copy_and_update(d: Dict[K, Any], patch: Dict[K, Any]) -> Dict[K, Any]:
  """Make a copy of d and apply the patch to a subset of fields."""
  ret = copy.copy(d)
  ret.update(patch)
  return ret


def remap_dict(d: Dict[K, V], keymap: Dict[K, K]) -> Dict[K, V]:
  """Return a (shallow) copy of d with some fields renamed.

  Keys which are not in keymap are left alone.

  Args:
    d: dict to rename
    keymap: map of old key -> new key

  Returns:
    new dict with fields renamed
  """
  return {keymap.get(k, k): d[k] for k in d}


def _extract_batch_length(preds):
  """Extracts batch length of predictions."""
  batch_length = None
  for key, value in preds.items():
    batch_length = batch_length or value.shape[0]
    if value.shape[0] != batch_length:
      raise ValueError('Batch length of predictions should be same. %s has '
                       'different batch length than others.' % key)
  return batch_length


def unbatch_preds(preds):
  """Unbatch predictions, as in estimator.predict().

  Args:
    preds: Dict[str, np.ndarray], where all arrays have the same first
      dimension.

  Yields:
    sequence of Dict[str, np.ndarray], with the same keys as preds.
  """
  if not isinstance(preds, dict):
    for pred in preds:
      yield pred
  else:
    for i in range(_extract_batch_length(preds)):
      yield {key: value[i] for key, value in preds.items()}
