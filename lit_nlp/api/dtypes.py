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
"""Dataclasses for representing structured output.

Classes in this file should be used for actual input/output data,
rather than in spec() metadata.

These classes can replace simple dicts or namedtuples, with two major
advantages:
- Type-checking (via pytype) doesn't work for dict fields, but does work for
  these dataclasses.
- Performance and memory use may be better, due to the use of __slots__

See the documentation for attr.s (https://www.attrs.org/) for more details.

Classes inheriting from DataTuple will be handled by serialize.py, and available
on the frontend as corresponding JavaScript objects.
"""
import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Text, Tuple, Union

import attr

JsonDict = Dict[Text, Any]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class DataTuple(metaclass=abc.ABCMeta):
  """Simple dataclasses.

  These are intended to be used for actual data, such as returned by
  dataset.examples and model.predict().

  Contrast with LitType and descendants, which are used in model and dataset
  /specs/ to represent types and metadata.
  """

  def to_json(self) -> JsonDict:
    """Used by serialize.py."""
    d = attr.asdict(self, recurse=False)
    d['__class__'] = 'DataTuple'
    d['__name__'] = self.__class__.__name__
    return d

  @staticmethod
  def from_json(d: JsonDict):
    """Used by serialize.py."""
    cls = globals()[d.pop('__name__')]  # class by name from this module
    return cls(**d)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SpanLabel(DataTuple):
  """Dataclass for individual span label preds. Can use this in model preds."""
  start: int  # inclusive
  end: int  # exclusive
  label: Optional[Text] = None
  align: Optional[Text] = None  # name of field (segment) this aligns to


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EdgeLabel(DataTuple):
  """Dataclass for individual edge label preds. Can use this in model preds."""
  span1: Tuple[int, int]  # inclusive, exclusive
  span2: Tuple[int, int]  # inclusive, exclusive
  label: Union[Text, int, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AnnotationCluster(DataTuple):
  """Dataclass for annotation clusters, which may span multiple segments."""
  label: Text
  spans: List[SpanLabel]
  score: Optional[float] = None

  def to_json(self) -> JsonDict:
    """Override serialization to properly convert nested objects."""
    d = super().to_json()
    d['spans'] = [s.to_json() for s in d['spans']]
    return d


# TODO(b/196886684): document API for salience interpreters.
@attr.s(auto_attribs=True, frozen=True, slots=True)
class TokenSalience(DataTuple):
  """Dataclass for a salience map over tokens."""
  tokens: List[str]
  salience: List[float]  # parallel to tokens


@attr.s(auto_attribs=True, frozen=True, slots=True)
class FeatureSalience(DataTuple):
  """Dataclass for a salience map over categorical and/or scalar features."""
  salience: Dict[str, float]


# TODO(b/196886684): document API for salience interpreters.
@attr.s(auto_attribs=True, frozen=True, slots=True)
class SequenceSalienceMap(DataTuple):
  """Dataclass for a salience map over a target sequence."""
  tokens_in: List[str]
  tokens_out: List[str]
  # <float>[num_tokens_out, num_tokens_in + num_tokens_out]
  salience: Sequence[Sequence[float]]  # usually, a np.ndarray


# LINT.IfChange
# pylint: disable=invalid-name
@attr.s(auto_attribs=True)
class LayoutSettings(DataTuple):
  hideToolbar: bool = False
  mainHeight: int = 45
  centerPage: bool = False


@attr.s(auto_attribs=True)
class LitComponentLayout(DataTuple):
  """Frontend UI layout (legacy); should match client/lib/types.ts."""
  # Keys are names of tabs; one must be called "Main".
  # Values are names of LitModule HTML elements,
  # e.g. data-table-module for the DataTableModule class.
  components: Dict[str, List[str]]
  layoutSettings: LayoutSettings = attr.ib(factory=LayoutSettings)
  description: Optional[str] = None

  def to_json(self) -> JsonDict:
    """Override serialization to properly convert nested objects."""
    # Not invertible, but these only go from server -> frontend anyway.
    return attr.asdict(self, recurse=True)


@attr.s(auto_attribs=True)
class LitCanonicalLayout(DataTuple):
  """Frontend UI layout; should match client/lib/types.ts."""
  # Keys are names of tabs, and values are names of LitModule HTML elements,
  # e.g. data-table-module for the DataTableModule class.
  upper: Dict[str, List[str]]
  lower: Dict[str, List[str]] = attr.ib(factory=dict)
  layoutSettings: LayoutSettings = attr.ib(factory=LayoutSettings)
  description: Optional[str] = None

  def to_json(self) -> JsonDict:
    """Override serialization to properly convert nested objects."""
    # Not invertible, but these only go from server -> frontend anyway.
    return attr.asdict(self, recurse=True)


LitComponentLayouts = Mapping[str, Union[LitComponentLayout,
                                         LitCanonicalLayout]]

# pylint: enable=invalid-name
# LINT.ThenChange(../client/lib/types.ts)
