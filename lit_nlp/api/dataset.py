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
import inspect
import random
from types import MappingProxyType  # pylint: disable=g-importing-member
from typing import List, Dict, Optional, Callable, Mapping, Sequence

from absl import logging

from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
ExampleId = types.ExampleId
Spec = types.Spec


class SliceWrapper(object):
  """Shim object to implement custom slicing via foo[a:b:c] rather than constructing a slice object explicitly."""

  def __init__(self, handler):
    self._handler = handler

  def __getitem__(self, slice_obj):
    return self._handler(slice_obj)


class Dataset(object):
  """Base class for LIT datasets.

  We recommend pre-loading the data in the constructor, but you can also stream
  on the fly in Dataset.examples() if desired.
  """

  _spec: Spec = {}
  _examples: List[JsonDict] = []
  _description: Optional[str] = None
  _base: Optional['Dataset'] = None

  def __init__(self,
               spec: Optional[Spec] = None,
               examples: Optional[List[JsonDict]] = None,
               description: Optional[str] = None,
               base: Optional['Dataset'] = None):
    """Base class constructor.

    This can derive from another dataset by passing the 'base' argument;
    if so it will pre-populate with those fields, and override only those
    specified individually as arguments.

    Args:
      spec: dataset spec
      examples: data examples (datapoints)
      description: optional human-readable description of this component
      base: optional base dataset to derive from
    """
    self._base = base
    if self._base is not None:
      self._examples = self._base.examples
      self._spec = self._base.spec()
      self._description = self._base.description()

    # Override from direct arguments.
    self._examples = examples or self._examples
    self._spec = spec or self._spec
    self._description = description or self._description

  def description(self) -> str:
    """Return a human-readable description of this component.

    Defaults to class docstring, but subclass may override this (or simply set
    self._description) to be instance-dependent - for example, including the
    path from which the data was loaded.

    Returns:
      (string) A human-readable description for display in the UI.
    """
    return self._description or inspect.getdoc(self) or ''  # pytype: disable=bad-return-type

  def spec(self) -> Spec:
    """Return a spec describing dataset elements."""
    return self._spec

  @property
  def examples(self) -> List[JsonDict]:
    """Return examples, in format described by spec."""
    return self._examples

  def __len__(self):
    return len(self.examples)

  @property
  def slice(self):
    """Syntactic sugar, allows dataset.slice[i:j] to return a new Dataset."""

    def _slicer(slice_obj):
      return Dataset(examples=self.examples[slice_obj], base=self)

    return SliceWrapper(_slicer)

  def sample(self, n, seed=42):
    """Return a new dataset with a random subset of examples."""
    rng = random.Random(seed)
    if n < len(self.examples):
      examples = rng.sample(self.examples, n)
    else:
      logging.warning(
          'Requested sample %d is larger than dataset size %d; returning full dataset.',
          n, len(self.examples))
      examples = list(self.examples)
    return Dataset(examples=examples, base=self)

  def shuffle(self, seed=42):
    """Return a new dataset with randomized example order."""
    # random.shuffle will shuffle in-place; use sample to make a new list.
    return self.sample(n=len(self), seed=seed)

  def remap(self, field_map: Dict[str, str]):
    """Return a copy of this dataset with some fields renamed."""
    new_spec = utils.remap_dict(self.spec(), field_map)
    new_examples = [utils.remap_dict(ex, field_map) for ex in self.examples]
    return Dataset(new_spec, new_examples, base=self)


IdFnType = Callable[[types.Input], ExampleId]


class IndexedDataset(Dataset):
  """Dataset with additional indexing information."""

  _index: Dict[ExampleId, IndexedInput] = {}

  def index_inputs(self, examples: List[types.Input]) -> List[IndexedInput]:
    """Create indexed versions of inputs."""
    return [
        IndexedInput({'data': example, 'id': self.id_fn(example), 'meta': {}})
        for example in examples
    ]  # pyformat: disable

  def __init__(self, *args, id_fn: IdFnType = None, **kw):
    super().__init__(*args, **kw)
    assert id_fn is not None, 'id_fn must be specified.'
    self.id_fn = id_fn
    self._indexed_examples = self.index_inputs(self._examples)
    self._index = {ex['id']: ex for ex in self._indexed_examples}

  @classmethod
  def index_all(cls, datasets: Mapping[str, Dataset], id_fn: IdFnType):
    """Convenience function to convert a dict of datasets."""
    return {name: cls(base=ds, id_fn=id_fn) for name, ds in datasets.items()}

  @property
  def indexed_examples(self) -> Sequence[IndexedInput]:
    return self._indexed_examples

  @property
  def index(self) -> Mapping[ExampleId, IndexedInput]:
    """Return a read-only view of the index."""
    return MappingProxyType(self._index)


class NoneDataset(Dataset):
  """Empty dataset, with fields as the union of model specs."""

  def __init__(self, models):  # pylint: disable=super-init-not-called
    self._examples = []
    self._models = models

  def spec(self):
    combined_spec = {}
    for _, model in self._models.items():
      req_inputs = {k: v for (k, v) in model.spec().input.items() if v.required}
      # Ensure that there are no conflicting spec keys.
      assert not self.has_conflicting_keys(combined_spec, req_inputs)
      combined_spec.update(req_inputs)

    return combined_spec

  def has_conflicting_keys(self, spec0: Spec, spec1: Spec):
    for k, v in spec0.items():
      if k in spec1 and spec1[k] != v:
        return True
    return False
