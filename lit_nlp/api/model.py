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
"""Base classes for LIT models."""
import abc
import inspect
import itertools
import multiprocessing  # for ThreadPool
from typing import List, Tuple, Iterable, Iterator, Text, Union

import attr
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec


def maybe_copy(arr):
  """Decide if we should make a copy of an array in order to release memory.

  NumPy arrays may be views into other array objects, by which a small array can
  maintain a persistent pointer to a large block of memory that prevents it from
  being garbage collected. This can quickly lead to memory issues as such
  blocks accumulate during inference.

  Args:
    arr: a NumPy array

  Returns:
    arr, or a copy of arr
  """
  if not isinstance(arr, np.ndarray):
    return arr
  # If this is not a view of another array.
  if arr.base is None:
    return arr
  # Heuristic to check if we should 'detach' this array from the parent blob.
  # We want to know if this array is a view that might leak memory.
  # The simplest check is if arr.base is larger than arr, but we don't want to
  # make unnecessary copies when this is just due to slicing along a batch,
  # because the other rows are likely still in use.
  # TODO(lit-dev): keep an eye on this, if we continue to have memory issues
  # we can make copies more aggressively.
  if arr.base.ndim > 1 and np.prod(arr.base.shape[1:]) > np.prod(arr.shape):
    return np.copy(arr)
  # If only a batch slice, reshape, or otherwise.
  return arr


def scrub_numpy_refs(output: JsonDict) -> JsonDict:
  """Scrub problematic pointers. See maybe_copy() and Model.predict()."""
  return {k: maybe_copy(v) for k, v in output.items()}


@attr.s(auto_attribs=True, frozen=True)
class ModelSpec(object):
  """Model spec."""
  input: Spec
  output: Spec

  def is_compatible_with_dataset(self, dataset_spec: Spec) -> bool:
    """Return true if this model is compatible with the dataset spec."""
    for key, field_spec in self.input.items():
      if key in dataset_spec:
        # If the field is in the dataset, make sure it's compatible.
        if not dataset_spec[key].is_compatible(field_spec):
          return False
      else:
        # If the field isn't in the dataset, only allow if the model marks as
        # optional.
        if field_spec.required:
          return False

    return True


class Model(metaclass=abc.ABCMeta):
  """Base class for LIT models."""

  def description(self) -> str:
    """Return a human-readable description of this component.

    Defaults to class docstring, but subclass may override this to be
    instance-dependent - for example, including the path from which the model
    was loaded.

    Returns:
      (string) A human-readable description for display in the UI.
    """
    return inspect.getdoc(self) or ''

  def max_minibatch_size(self) -> int:
    """Maximum minibatch size for this model."""
    return 1

  @abc.abstractmethod
  def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
    """Run prediction on a batch of inputs.

    Args:
      inputs: sequence of inputs, following model.input_spec()

    Returns:
      list of outputs, following model.output_spec()
    """
    return

  def load(self, path: str):
    """Load and return a new instance of this model loaded from a new path.

    By default this method does nothing. Models can override this method in
    order to allow dynamic model loading in LIT through the UI. Models
    overriding this method should use the provided path string and create and
    return a new instance of its model class.

    Args:
      path: The path to the persisted model information, used in model's
      construction.

    Returns:
      (Model) A model loaded with information from the provided path.
    """
    del path
    raise NotImplementedError('Model has no load method defined for dynamic '
                              'loading')

  @abc.abstractmethod
  def input_spec(self) -> types.Spec:
    """Return a spec describing model inputs."""
    return

  @abc.abstractmethod
  def output_spec(self) -> types.Spec:
    """Return a spec describing model outputs."""
    return

  # TODO(lit-dev): annotate as @final once we migrate to python 3.8+
  def spec(self) -> ModelSpec:
    return ModelSpec(input=self.input_spec(), output=self.output_spec())

  def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
    """Return the full vocabulary and embedding table.

    Implementing this is optional, but needed for some techniques such as
    HotFlip which use the embedding table to search over candidate words.

    Returns:
      (<string>[vocab_size], <float32>[vocab_size, emb_dim])
    """
    raise NotImplementedError('get_embedding_table() not implemented for ' +
                              self.__class__.__name__)

  def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
    """For internal use by UMAP and other sklearn-based models."""
    raise NotImplementedError(
        'fit_transform_with_metadata() not implemented for ' +
        self.__class__.__name__)

  ##
  # Concrete implementations of common functions.
  def predict(self,
              inputs: Iterable[JsonDict],
              scrub_arrays=True,
              **kw) -> Iterator[JsonDict]:
    """Run prediction on a dataset.

    This uses minibatch inference for efficiency, but yields per-example output.

    Args:
      inputs: iterable of input dicts
      scrub_arrays: if True, will copy some returned NumPy arrays in order to
        allow garbage collection of intermediate data. Strongly recommended if
        results will not be immediately consumed and discarded, as otherwise the
        common practice of slicing arrays returned by e.g. TensorFlow can result
        in large memory leaks.
      **kw: additional kwargs passed to predict_minibatch()

    Returns:
      model outputs, for each input
    """
    results = self._batched_predict(inputs, **kw)
    if scrub_arrays:
      results = (scrub_numpy_refs(res) for res in results)
    return results

  def _batched_predict(self, inputs: Iterable[JsonDict],
                       **kw) -> Iterator[JsonDict]:
    """Internal helper to predict using minibatches."""
    minibatch_size = self.max_minibatch_size(**kw)
    minibatch = []
    for ex in inputs:
      if len(minibatch) < minibatch_size:
        minibatch.append(ex)
      if len(minibatch) >= minibatch_size:
        yield from self.predict_minibatch(minibatch, **kw)
        minibatch = []
    if len(minibatch) > 0:  # pylint: disable=g-explicit-length-test
      yield from self.predict_minibatch(minibatch, **kw)

  # TODO(b/171513556): remove this method.
  def predict_with_metadata(self, indexed_inputs: Iterable[JsonDict],
                            **kw) -> Iterator[JsonDict]:
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex['data'] for ex in indexed_inputs), **kw)


class ModelWrapper(Model):
  """Wrapper for a LIT model.

  This class acts as an identity function, with pass-through implementations of
  the Model API. Subclasses of this can implement only those methods that need
  to be modified.
  """

  def __init__(self, model: Model):
    self._wrapped = model

  @property
  def wrapped(self):
    """Access the wrapped model."""
    return self._wrapped

  def description(self) -> str:
    return self.wrapped.description()

  def max_minibatch_size(self) -> int:
    return self.wrapped.max_minibatch_size()

  def predict_minibatch(self, inputs: List[JsonDict], **kw) -> List[JsonDict]:
    return self.wrapped.predict_minibatch(inputs, **kw)

  def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterator[JsonDict]:
    return self.wrapped.predict(inputs, **kw)

  # NOTE: if a subclass modifies predict(), it should also override this to
  # call the custom predict() method - otherwise this will delegate to the
  # wrapped class and call /that class's/ predict() method, likely leading to
  # incorrect results.
  # b/171513556 will solve this problem by removing the need for any
  # *_with_metadata() methods.
  def predict_with_metadata(self, indexed_inputs: Iterable[JsonDict],
                            **kw) -> Iterator[JsonDict]:
    return self.wrapped.predict_with_metadata(indexed_inputs, **kw)

  def load(self, path: str):
    """Load a new model and wrap it with this class."""
    new_model = self.wrapped.load(path)
    return self.__class__(new_model)

  def input_spec(self) -> types.Spec:
    return self.wrapped.input_spec()

  def output_spec(self) -> types.Spec:
    return self.wrapped.output_spec()

  def spec(self) -> ModelSpec:
    return ModelSpec(input=self.input_spec(), output=self.output_spec())

  ##
  # Special methods
  def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
    return self.wrapped.get_embedding_table()

  def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
    return self.wrapped.fit_transform_with_metadata(indexed_inputs)


class BatchedRemoteModel(Model):
  """Generic base class for remotely-hosted models.

  Implements concurrent request batching; subclass need only implement
  predict_minibatch() and max_minibatch_size().

  If subclass overrides __init__, it should be sure to call super().__init__()
  to set up the threadpool.
  """

  def __init__(self,
               max_concurrent_requests: int = 4,
               max_qps: Union[int, float] = 25):
    # Use a local thread pool for concurrent requests, so we can keep the server
    # busy during network transit time and local pre/post-processing.
    self._max_qps = max_qps
    self._pool = multiprocessing.pool.ThreadPool(max_concurrent_requests)

  def predict(self, inputs: Iterable[JsonDict], **kw) -> Iterator[JsonDict]:
    batches = utils.batch_iterator(
        inputs, max_batch_size=self.max_minibatch_size())
    batches = utils.rate_limit(batches, self._max_qps)
    pred_batches = self._pool.imap(self.predict_minibatch, batches)
    return itertools.chain.from_iterable(pred_batches)

  def max_minibatch_size(self) -> int:
    """Maximum minibatch size for this model. Subclass can override this."""
    return 1

  @abc.abstractmethod
  def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
    """Run prediction on a batch of inputs.

    Subclass should implement this.

    Args:
      inputs: sequence of inputs, following model.input_spec()

    Returns:
      list of outputs, following model.output_spec()
    """
    return
