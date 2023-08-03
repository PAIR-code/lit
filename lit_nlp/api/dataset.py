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
"""Base classes for LIT models."""
from collections.abc import Callable, Mapping, Sequence
import hashlib
import glob
import inspect
import os
import random
import types
from typing import Optional, Union, cast

from absl import logging
from lit_nlp.api import types as lit_types
from lit_nlp.lib import serialize
from lit_nlp.lib import utils

ExampleId = lit_types.ExampleId
IdFnType = Callable[[lit_types.JsonDict], lit_types.ExampleId]
IndexedInput = lit_types.IndexedInput
JsonDict = lit_types.JsonDict
Spec = lit_types.Spec

LIT_FILE_EXTENSION = '.lit.jsonl'
LIT_SPEC_EXTENSION = '.spec'

INPUT_ID_FIELD = '_id'
INPUT_META_FIELD = '_meta'
INPUT_INTERNAL_FIELDS = (INPUT_ID_FIELD, INPUT_META_FIELD)


# This is used here and in caching.py, but we define here to avoid a circular
# dependency of dataset -> caching -> model -> dataset
def input_hash(example: lit_types.JsonDict) -> lit_types.ExampleId:
  """Create stable hash of an input example."""
  raw_example = {
      k: v for k, v in example.items()
      if k not in INPUT_INTERNAL_FIELDS
  }
  json_str = serialize.to_json(raw_example, simple=True, sort_keys=True)
  return lit_types.ExampleId(hashlib.md5(json_str.encode('utf-8')).hexdigest())


def write_examples(examples: Sequence[JsonDict], path: str):
  """Write examples to disk as LIT JSONL format."""
  with open(path, 'w') as fd:
    for ex in examples:
      fd.write(serialize.to_json(ex) + '\n')


def write_spec(spec: Spec, path: str):
  """Write spec to disk as LIT JSON format."""
  with open(path, 'w') as fd:
    fd.write(serialize.to_json(spec, indent=2))


class SliceWrapper(object):
  """Shim object to implement custom slicing via foo[a:b:c] rather than constructing a slice object explicitly."""

  def __init__(self, handler):
    self._handler = handler

  def __getitem__(self, slice_obj):
    return self._handler(slice_obj)


class Dataset(object):
  """Base class for LIT datasets."""

  _spec: Spec = {}
  _examples: list[JsonDict] = []
  _description: Optional[str] = None
  _base: Optional['Dataset'] = None

  def __init__(self,
               spec: Optional[Spec] = None,
               examples: Optional[list[JsonDict]] = None,
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
      # In case user child class requires the instance to convert examples
      # this makes sure the user class is preserved. We cannot do this below
      # as the default method is static and does not require instance.
      self.bytes_from_lit_example = self._base.bytes_from_lit_example
      self.lit_example_from_bytes = self._base.lit_example_from_bytes

    # Override from direct arguments.
    self._examples = examples if examples is not None else self._examples
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

  @classmethod
  def init_spec(cls) -> Optional[lit_types.Spec]:
    """Attempts to infer a Spec describing a Dataset's constructor parameters.

    The Dataset base class attempts to infer a Spec for the constructor using
    `lit_nlp.api.types.infer_spec_for_func()`.

    If successful, this function will return a `dict[str, LitType]`. If
    unsucessful (i.e., the inferencer raises a `TypeError` because it encounters
    a parameter that it not supported by `infer_spec_for_func()`), this function
    will return None, log a warning describing where and how the inferencing
    failed, and LIT users **will not** be able to load new instances of this
    Dataset from the UI.

    Returns:
      A Spec representation of the Dataset's constructor, or None if a Spec
      could not be inferred.
    """
    try:
      spec = lit_types.infer_spec_for_func(cls.__init__)
    except TypeError as e:
      spec = None
      logging.warning(
          "Unable to infer init spec for dataset '%s'. %s", cls.__name__, str(e)
      )
    return spec

  def load(self, path: str):
    """Load and return additional previously-saved datapoints for this dataset.

    Args:
      path: The path to the persisted datapoint file.

    Returns:
      (Dataset) A dataset containing the loaded data.
    """
    if self._base is not None:
      return self._base.load(path)
    pass

  def save(self, examples: list[IndexedInput], path: str):
    """Save newly-created datapoints to disk in a dataset-specific format.

    Subclasses should override this method if they wish to save new, persisted
    datapoints in their own file format in addition to the LIT-specific format
    they are already saved in.

    Args:
      examples: A list of datapoints to save.
      path: The path to save the datapoints to.

    Returns:
      (string) The path to the saved data, or None if unimplemented.
    """
    if self._base is not None:
      return self._base.save(examples, path)
    pass

  def spec(self) -> Spec:
    """Return a spec describing dataset elements."""
    return self._spec

  @property
  def examples(self) -> list[JsonDict]:
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
          'Requested sample %d is larger than dataset size %d; returning full'
          ' dataset.',
          n,
          len(self.examples),
      )
      examples = list(self.examples)
    return Dataset(examples=examples, base=self)

  def filter(self, predicate: Callable[[JsonDict], bool]):
    selected_examples = list(filter(predicate, self.examples))
    return Dataset(examples=selected_examples, base=self)

  def shuffle(self, seed=42):
    """Return a new dataset with randomized example order."""
    # random.shuffle will shuffle in-place; use sample to make a new list.
    return self.sample(n=len(self), seed=seed)

  def remap(self, field_map: Mapping[str, str]):
    """Return a copy of this dataset with some fields renamed."""
    new_spec = utils.remap_dict(self.spec(), field_map)
    new_examples = [utils.remap_dict(ex, field_map) for ex in self.examples]
    return Dataset(new_spec, new_examples, base=self)

  @staticmethod
  def lit_example_from_bytes(input_bytes: bytes) -> Optional[JsonDict]:
    """Convert bytes representation to LIT example."""
    return serialize.from_json(input_bytes.decode('utf-8'))

  @staticmethod
  def bytes_from_lit_example(lit_example: JsonDict) -> bytes:
    """Convert LIT example to bytes representation."""
    return serialize.to_json(lit_example).encode('utf-8')


class IndexedDataset(Dataset):
  """Dataset with additional indexing information."""

  _index: dict[ExampleId, IndexedInput] = {}

  def _normalize_example(
      self, data: JsonDict, ex_id: ExampleId, meta: lit_types.InputMetadata
  ):
    return types.MappingProxyType(dict(data, _id=ex_id, _meta=meta))

  def index_inputs(
      self, examples: list[lit_types.JsonDict]
  ) -> list[IndexedInput]:
    """Create indexed versions of inputs."""
    indexed = []
    for example in examples:
      ex_id = example.get(INPUT_ID_FIELD, self.id_fn(example))
      ex_meta = example.get(
          INPUT_META_FIELD,
          lit_types.InputMetadata(added=None, parentId=None, source=None),
      )
      indexed.append(
          IndexedInput(
              data=types.MappingProxyType(
                  example | {INPUT_ID_FIELD: ex_id, INPUT_META_FIELD: ex_meta}
              ),
              id=ex_id,
              meta=ex_meta,
          )
      )
    return indexed

  def __init__(
      self,
      *args,
      id_fn: Optional[IdFnType] = None,
      indexed_examples: Optional[list[IndexedInput]] = None,
      **kw,
  ):
    # The base Dataset class will initialize self._examples in this call to
    # super().__init__(), which may or may not include the _id and _meta fields.
    super().__init__(*args, **kw)
    self.id_fn = id_fn if id_fn is not None else input_hash

    if indexed_examples:
      self._indexed_examples = indexed_examples
      # Ensure that all indexed exampls provide a readonly view of their data.
      for ie in self._indexed_examples:
        if not isinstance((ie_data := ie['data']), types.MappingProxyType):
          ie['data'] = self._normalize_example(ie_data, ie['id'], ie['meta'])
    else:
      self._indexed_examples = self.index_inputs(self._examples)

    self._examples = [
        self._normalize_example(ex['data'], ex['id'], ex.get('meta', {}))
        for ex in self._indexed_examples
    ]
    self._index = {ex['id']: ex for ex in self._indexed_examples}

  @property
  def slice(self):
    """Syntactic sugar, allows .slice[i:j] to return a new IndexedDataset."""

    def _slicer(slice_obj):
      return IndexedDataset(
          indexed_examples=self.indexed_examples[slice_obj],
          id_fn=self.id_fn,
          base=self
      )

    return SliceWrapper(_slicer)

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
    return types.MappingProxyType(self._index)

  def save(self, examples: list[IndexedInput], path: str):
    """Save newly-created datapoints to disk.

    Args:
      examples: A list of datapoints to save.
      path: The path to save the datapoints to.

    Returns:
      (string) The file path of the saved datapoints.
    """
    # Attempt to save the datapoints using the base save method, which
    # datasets can override. Then also save in the lit json format and save
    # the spec as well.
    if not path.endswith(LIT_FILE_EXTENSION):
      if (base_dataset := self._base) is not None:
        base_dataset.save(examples, path)
      path += LIT_FILE_EXTENSION

    write_examples(examples, path)
    write_spec(self.spec(), path + LIT_SPEC_EXTENSION)

    return path

  def load(self, path: str):
    """Load and return additional previously-saved datapoints for this dataset.

    Args:
      path: The path to the persisted datapoint file.

    Returns:
      (IndexedDataset) A dataset containing the loaded data.

    """
    if not path.endswith(LIT_FILE_EXTENSION):
      # Try to load data using the base load method. If any data is
      # returned, then use that. Otherwise try loading the lit json extension
      # data format.
      base_dataset = self._base
      new_dataset = base_dataset.load(path) if base_dataset else None

      if new_dataset is not None:
        description = (f'{len(new_dataset)} examples from '
                       f'{path}\n{self._base.description()}')
        return IndexedDataset(
            base=new_dataset, id_fn=self.id_fn, description=description)

      path += LIT_FILE_EXTENSION

    with open(path, 'r') as fd:
      examples = [
          cast(IndexedInput, serialize.from_json(line))
          for line in fd.readlines()
      ]

    # Load the side-by-side spec if it exists on disk.
    spec_path = path + LIT_SPEC_EXTENSION
    if os.path.exists(spec_path):
      with open(spec_path, 'r') as fd:
        spec = serialize.from_json(fd.read())
    else:
      spec = None

    description = f'{len(examples)} examples from {path}'
    if self._base is not None:
      description += '\n' + self._base.description()

    return IndexedDataset(
        base=self._base,
        indexed_examples=examples,
        spec=spec,
        description=description,
        id_fn=self.id_fn)

  def __hash__(self):
    return hash(tuple([ex['id'] for ex in self._indexed_examples]))

  def __eq__(self, other):
    self_ids = [ex['id'] for ex in self._indexed_examples]
    other_ids = [ex['id'] for ex in other._indexed_examples]
    return self_ids == other_ids


def load_lit_format(
    path: str, *args, id_fn=input_hash, **kw
) -> Union[Dataset, IndexedDataset]:
  """Load data from LIT jsonl format."""
  with open(path + LIT_SPEC_EXTENSION, 'r') as fd:
    spec = serialize.from_json(fd.read())

  with open(path, 'r') as fd:
    examples = [serialize.from_json(line) for line in fd.readlines()]

  first_example_keys = set(ex.keys() if (ex := examples[0]) else [])
  # TODO(b/294233896): remove this once input representations are consolidated.
  if first_example_keys.issuperset({'id', 'data'}):
    return IndexedDataset(
        spec=spec,
        indexed_examples=cast(list[lit_types.IndexedInput], examples),
        id_fn=id_fn,
        *args,
        **kw,
    )
  else:
    return Dataset(spec=spec, examples=examples, *args, **kw)


# TODO(b/202210900): Remove "NoneDataset" once the LIT front-end constructs its
# own "NoneDataset" equivalent.
class NoneDataset(Dataset):
  """Empty dataset, with fields as the union of model specs."""

  def __init__(self, models):  # pylint: disable=super-init-not-called
    self._examples = []
    self._models = models

  def spec(self):
    combined_spec = {}
    for _, model in self._models.items():
      req_inputs = {k: v for (k, v) in model.input_spec().items() if v.required}
      combined_spec = utils.combine_specs(combined_spec, req_inputs)

    return combined_spec
