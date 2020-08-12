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
import random
from typing import List, Dict

from absl import logging

from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict
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

  def __init__(self, spec, examples):
    self._spec = spec
    self._examples = examples

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
    slicer = lambda slice_obj: Dataset(self.spec(), self.examples[slice_obj])
    return SliceWrapper(slicer)

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
    return Dataset(self.spec(), examples)

  def shuffle(self, seed=42):
    """Return a new dataset with randomized example order."""
    # random.shuffle will shuffle in-place; use sample to make a new list.
    return self.sample(n=len(self), seed=seed)

  def remap(self, field_map: Dict[str, str]):
    """Return a copy of this dataset with some fields renamed."""
    new_spec = utils.remap_dict(self.spec(), field_map)
    new_examples = [utils.remap_dict(ex, field_map) for ex in self.examples]
    return Dataset(new_spec, new_examples)
