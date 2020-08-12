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
from typing import List, Tuple, Iterable, Iterator, Text

import attr
from lit_nlp.api import types
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
        # If the field is in the dataset, make sure it matches.
        if not field_spec.is_compatible(dataset_spec[key]):
          return False
      else:
        # If the field isn't in the dataset, only allow if the model marks as
        # optional.
        if field_spec.required:
          return False

    return True


class Model(metaclass=abc.ABCMeta):
  """Base class for LIT models."""

  # pylint: disable=unused-argument
  def max_minibatch_size(self, config=None) -> int:
    """Maximum minibatch size for this model."""
    return 1

  # pylint: enable=unused-argument

  @abc.abstractmethod
  def predict_minibatch(self,
                        inputs: List[JsonDict],
                        config=None) -> List[JsonDict]:
    """Run prediction on a batch of inputs.

    TODO(lit-team): add a 'level' argument (enumerate preprocessing, forward,
    backward) to save on expensive operations when not needed. For example,
    computing gradients invokes a backward pass, which costs ~2x the memory and
    ~2x the compute of just running a forward prediction.

    Args:
      inputs: sequence of inputs, following model.input_spec()
      config: (optional) predict-time model config (beam size, num candidates,
        etc.)

    Returns:
      list of outputs, following model.output_spec()
    """
    return

  @abc.abstractmethod
  def input_spec(self) -> types.Spec:
    """Return a spec describing model inputs."""
    return

  @abc.abstractmethod
  def output_spec(self) -> types.Spec:
    """Return a spec describing model outputs."""
    return

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
  def predict_single(self, one_input: JsonDict, **kw) -> JsonDict:
    """Run prediction on a single input."""
    return self.predict_minibatch([one_input], **kw)[0]

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

  def predict_with_metadata(self, indexed_inputs: Iterable[JsonDict],
                            **kw) -> Iterator[JsonDict]:
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex['data'] for ex in indexed_inputs), **kw)
