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
import itertools
import queue
import threading
import time

from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, TypeVar, Union

T = TypeVar('T')
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


def rate_limit(iterable, qps: Union[int, float]):
  """Rate limit an iterator."""
  for item in iterable:
    yield item
    time.sleep(1.0 / qps)


def batch_iterator(items: Iterable[T],
                   max_batch_size: int) -> Iterator[List[T]]:
  """Create batches from an input stream.

  Use this to create batches, e.g. to feed to a model.
  The output can be easily flattened again using itertools.chain.from_iterable.

  Args:
    items: stream of items
    max_batch_size: maximum size of resulting batches

  Yields:
    batches of size <= max_batch_size
  """
  minibatch = []
  for item in items:
    if len(minibatch) < max_batch_size:
      minibatch.append(item)
    if len(minibatch) >= max_batch_size:
      yield minibatch
      minibatch = []
  if len(minibatch) > 0:  # pylint: disable=g-explicit-length-test
    yield minibatch


def batch_inputs(input_records: Sequence[Dict[K, V]]) -> Dict[K, List[V]]:
  """Batch inputs from list-of-dicts to dict-of-lists."""
  assert input_records, 'Must have non-empty batch!'
  ret = {}
  for k in input_records[0]:
    ret[k] = [r[k] for r in input_records]
  return ret


def _extract_batch_length(preds):
  """Extracts batch length of predictions."""
  batch_length = None
  for key, value in preds.items():
    this_length = (
        len(value) if isinstance(value, (list, tuple)) else value.shape[0])
    batch_length = batch_length or this_length
    if this_length != batch_length:
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


def find_all_combinations(l: List[Any], min_element_count: int,
                          max_element_count: int) -> List[List[Any]]:
  """Finds all possible ways how elements of a list can be combined.

  E.g., all combinations of list [1, 2, 3] are
  [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]].

  Args:
    l: a list of arbitrary elements.
    min_element_count: the minimum number of elements that every combination
      should contain.
    max_element_count: the maximum number of elements that every combination
      should contain.

  Returns:
    The list of all possible combinations given the constraints.
  """
  result: List[List[Any]] = []
  min_element_count = max(1, min_element_count)
  max_element_count = min(max_element_count, len(l))
  for element_count in range(min_element_count, max_element_count + 1):
    result.extend(list(x) for x in itertools.combinations(l, element_count))
  return result


class TaskQueue(queue.Queue):
  """A simple task queue for processing jobs in a thread pool."""

  def __init__(self, num_workers=1):
    # TODO(lit-dev): Could use QueueHandler and QueueListener for this.
    queue.Queue.__init__(self)
    self.num_workers = num_workers
    self.start_workers()

  def add_task(self, task, *args, **kwargs):
    args = args or ()
    kwargs = kwargs or {}
    self.put((task, args, kwargs))

  def start_workers(self):
    for _ in range(self.num_workers):
      t = threading.Thread(target=self.worker)
      t.daemon = True
      t.start()

  def worker(self):
    while True:
      item, args, kwargs = self.get()
      item(*args, **kwargs)
      self.task_done()
