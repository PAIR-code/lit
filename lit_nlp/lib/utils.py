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

from collections.abc import Collection
import itertools
import queue
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, Union
import uuid

from lit_nlp.api import types as lit_types
import numpy as np

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


def find_keys(d: Mapping[K, V], predicate: Callable[[V], bool]) -> list[K]:
  """Find keys where values match predicate."""
  return [k for k, v in d.items() if predicate(v)]


def find_spec_keys(d: Mapping[K, Any], types) -> list[K]:
  """Find keys where values match one or more types."""
  return find_keys(d, lambda v: isinstance(v, types))


def filter_by_keys(
    d: Mapping[K, V], predicate: Callable[[K], bool]
) -> dict[K, V]:
  """Filter to keys matching predicate."""
  return {k: v for k, v in d.items() if predicate(k)}


def spec_contains(d: dict[str, Any], types) -> bool:
  """Returns true if the spec contains any field with one of these types."""
  return bool(find_spec_keys(d, types))


def remap_dict(d: Mapping[K, V], keymap: Mapping[K, K]) -> dict[K, V]:
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


def batch_iterator(
    items: Iterable[T], max_batch_size: int
) -> Iterator[list[T]]:
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


def batch_inputs(
    input_records: Sequence[Mapping[K, V]], keys: Optional[Collection[K]] = None
) -> dict[K, list[V]]:
  """Batch inputs from list-of-dicts to dict-of-lists."""
  assert input_records, 'Must have non-empty batch!'
  if keys is None:
    keys = input_records[0].keys()

  ret = {}
  for k in keys:
    ret[k] = [r[k] for r in input_records]
  return ret


def _extract_batch_length(preds):
  """Extracts batch length of predictions."""
  batch_length = None
  for key, value in preds.items():
    this_length = (
        len(value) if isinstance(value, (list, tuple)) else value.shape[0]
    )
    batch_length = batch_length or this_length
    if this_length != batch_length:
      raise ValueError('Batch length of predictions should be same. %s has '
                       'different batch length than others.' % key)
  return batch_length


def unbatch_preds(
    preds: Mapping[K, Sequence[V]] | Sequence[dict[K, V]]
) -> Iterable[dict[K, V]]:
  """Unbatch predictions, as in estimator.predict().

  Args:
    preds: dict[str, np.ndarray], where all arrays have the same first
      dimension.

  Yields:
    sequence of dict[str, np.ndarray], with the same keys as preds.
  """
  if not isinstance(preds, dict):
    for pred in preds:
      yield pred
  else:
    for i in range(_extract_batch_length(preds)):
      yield {key: value[i] for key, value in preds.items()}


def pad1d(arr: list[T], min_len: int, pad_val: T) -> list[T]:
  """Pad a list to the target length."""
  return arr + [pad_val] * max(0, min_len - len(arr))


def find_all_combinations(
    l: list[Any], min_element_count: int, max_element_count: int
) -> list[list[Any]]:
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
  result: list[list[Any]] = []
  min_element_count = max(1, min_element_count)
  max_element_count = min(max_element_count, len(l))
  for element_count in range(min_element_count, max_element_count + 1):
    result.extend(list(x) for x in itertools.combinations(l, element_count))
  return result


def coerce_real(vals: np.ndarray, limit=0.0001):
  """Return a copy of the array with only the real numbers, with a check.

  If any of the imaginary part of a value is greater than the provided limit,
  then assert an error.

  Args:
    vals: The array to convert
    limit: The limit above which any imaginary part of a value causes an error.

  Returns:
    The array with only the real portions of the numbers.
  """
  assert np.all(np.imag(vals) < limit), (
      'Array contains imaginary part out of acceptable limits.')
  return np.real(vals)


def get_uuid():
  """Return a randomly-generated UUID hex string."""
  return uuid.uuid4().hex


def validate_config_against_spec(
    config: lit_types.JsonDict,
    spec: lit_types.Spec,
    name: str,
    raise_for_unsupported: bool = False,
):
  """Validates that the provided config is compatible with the Spec.

  Args:
    config: The configuration parameters, such as extracted from the data of an
      HTTP Request, that are to be used in a function call.
    spec: A Spec defining the shape of allowed configuration parameters for the
      associated LIT component.
    name: The name of the endpoint, interpreter, etc. providing the Spec against
      which the config is valdiated.
    raise_for_unsupported: If true, raises a KeyError if the config contains
      keys that are not present in the Spec. Unsupported keys are assumed to be
      acceptable for subclasses of lit_nlp.api.components, but unacceptable for
      APIs that instantiate new instances of a class (e.g., /create_dataset).

  Returns:
    The config passed in as the first argument, if validation is successful.

  Raises:
    KeyError: Under two conditions: 1) the `config` is missing one or more
      required fields defined in the `spec`, or 2) the `config` contains fields
      not defined in the `spec`. Either of these conditions would likely result
      in a TypeError (for missing or unexpected arguments) if the `config` was
      used in a call.
  """
  missing_required_keys = [
      param_name for param_name, param_type in spec.items()
      if param_type.required and param_name not in config
  ]
  if missing_required_keys:
    raise KeyError(f'{name} missing required params: {missing_required_keys}')

  unsupported_keys = [
      param_name for param_name in config
      if param_name not in spec
  ]
  if raise_for_unsupported and unsupported_keys:
    raise KeyError(f'{name} received unsupported params: {unsupported_keys}')

  return config


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
