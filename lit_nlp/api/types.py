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
"""Type classes for LIT inputs and outputs.

These are simple dataclasses used in model.input_spec() and model.output_spec()
to describe the semantics of the model outputs, while allowing clients to still
use flexible data structures.

These are used by the LIT framework to configure front-end components and to
enable different generation and visualization modules. For example, the input
spec allows LIT to automatically generate input forms for common types like text
segments or class labels, while the output spec describes how the model output
should be rendered.
"""
import abc
from typing import Any, Dict, List, NewType, Optional, Sequence, Text, Tuple, Union

import attr

JsonDict = Dict[Text, Any]
Input = JsonDict  # TODO(lit-dev): stronger typing using NewType
IndexedInput = NewType("IndexedInput", JsonDict)  # has keys: id, data, meta
ExampleId = Text
TokenTopKPredsList = List[List[Tuple[str, float]]]


##
# Base classes, for common functionality and type grouping.
@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class LitType(metaclass=abc.ABCMeta):
  """Base class for LIT Types."""
  required: bool = True  # for input fields, mark if required by the model.
  annotated: bool = False  # If this type is created from an Annotator.
  # TODO(lit-dev): Add defaults for all LitTypes
  default = None  # an optional default value for a given type.

  def is_compatible(self, other):
    """Check equality, ignoring some fields."""
    # We allow this class to be a subclass of the other.
    if not isinstance(self, type(other)):
      return False
    d1 = attr.asdict(self)
    d1.pop("required", None)
    d2 = attr.asdict(other)
    d2.pop("required", None)
    return d1 == d2

  def to_json(self) -> JsonDict:
    """Used by serialize.py."""
    d = attr.asdict(self)
    d["__class__"] = "LitType"
    d["__name__"] = self.__class__.__name__
    # All parent classes, from method resolution order (mro).
    # Use this to check inheritance on the frontend.
    d["__mro__"] = [a.__name__ for a in self.__class__.__mro__]
    return d

  @staticmethod
  def from_json(d: JsonDict):
    """Used by serialize.py."""
    cls = globals()[d.pop("__name__")]  # class by name from this module
    del d["__mro__"]
    return cls(**d)


Spec = Dict[Text, LitType]

# Attributes that should be treated as a reference to other fields.
FIELD_REF_ATTRIBUTES = frozenset(
    {"parent", "align", "align_in", "align_out", "grad_for"})


def _remap_leaf(leaf: LitType, keymap: Dict[str, str]) -> LitType:
  """Remap any field references on a LitType."""
  d = attr.asdict(leaf)  # mutable
  d = {
      k: (keymap.get(v, v) if k in FIELD_REF_ATTRIBUTES else v)
      for k, v in d.items()
  }
  return leaf.__class__(**d)


def remap_spec(spec: Spec, keymap: Dict[str, str]) -> Spec:
  """Rename fields in a spec, with a best-effort to also remap field references."""
  ret = {}
  for k, v in spec.items():
    new_key = keymap.get(k, k)
    new_value = _remap_leaf(v, keymap)
    ret[new_key] = new_value
  return ret


##
# Concrete type clases
# LINT.IfChange


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class String(LitType):
  """User-editable text input.

  All automated edits are disabled for this type.

  Mainly used for string inputs that have special formatting, and should only
  be edited manually.
  """
  default: Text = ""


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TextSegment(LitType):
  """Text input (untokenized), a single string."""
  default: Text = ""


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageBytes(LitType):
  """An image, an encoded base64 ascii string (starts with 'data:image...')."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedText(TextSegment):
  """Generated (untokenized) text."""
  # Name of a TextSegment field to evaluate against
  parent: Optional[Text] = None


ScoredTextCandidates = List[Tuple[str, Optional[float]]]


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedTextCandidates(TextSegment):
  """Multiple candidates for GeneratedText; values are List[(text, score)]."""
  # Name of a TextSegment field to evaluate against
  parent: Optional[Text] = None

  @staticmethod
  def top_text(value: ScoredTextCandidates) -> str:
    return value[0][0] if len(value) else ""


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ReferenceTexts(LitType):
  """Multiple candidates for TextSegment; values are List[(text, score)]."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class URL(TextSegment):
  """TextSegment that should be interpreted as a URL."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SearchQuery(TextSegment):
  """TextSegment that should be interpreted as a search query."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Tokens(LitType):
  """Tokenized text, as List[str]."""
  default: List[Text] = attr.Factory(list)
  # Name of a TextSegment field from the input
  # TODO(b/167617375): should we use 'align' here?
  parent: Optional[Text] = None
  mask_token: Optional[Text] = None  # optional mask token for input


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenTopKPreds(LitType):
  """Predicted tokens, as from a language model.

  Data should be a List[List[Tuple[str, float]]], where the inner list contains
  (word, probability) in descending order.
  """
  align: Text = None  # name of a Tokens field in the model output
  parent: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Scalar(LitType):
  """Scalar value, a single float or int."""
  min_val: float = 0
  max_val: float = 1
  default: float = 0
  step: float = .01


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class RegressionScore(Scalar):
  """Regression score, a single float."""
  # name of a Scalar or RegressionScore field in input
  parent: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ReferenceScores(LitType):
  """Score of one or more target sequences, as List[float]."""
  # name of a TextSegment or ReferenceTexts field in the input
  parent: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CategoryLabel(LitType):
  """Category or class label, a single string."""
  # Optional vocabulary to specify allowed values.
  # If omitted, any value is accepted.
  vocab: Optional[Sequence[Text]] = None  # label names


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MulticlassPreds(LitType):
  """Multiclass predicted probabilities, as <float>[num_labels]."""
  # Vocabulary is required here for decoding model output.
  # Usually this will match the vocabulary in the corresponding label field.
  vocab: Sequence[Text]  # label names
  null_idx: Optional[int] = None  # vocab index of negative (null) label
  parent: Optional[Text] = None  # CategoryLabel field in input
  autosort: Optional[bool] = False  # Enable automatic sorting

  @property
  def num_labels(self):
    return len(self.vocab)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SequenceTags(LitType):
  """Sequence tags, aligned to tokens.

  The data should be a list of string labels, one for each token.
  """
  align: Text  # name of Tokens field


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SpanLabels(LitType):
  """Span labels, a List[dtypes.SpanLabel] aligned to tokens.

  Span labels can cover more than one token, may not cover all tokens in the
  sentence, and may overlap with each other.
  """
  align: Text  # name of Tokens field
  parent: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class EdgeLabels(LitType):
  """Edge labels, a List[dtypes.EdgeLabel] between pairs of spans.

  This is a general form for structured prediction output; each entry consists
  of (span1, span2, label). See
  https://arxiv.org/abs/1905.06316 (Tenney et al. 2019) and
  https://github.com/nyu-mll/jiant/tree/master/probing#data-format for more
  details.
  """
  align: Text  # name of Tokens field


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MultiSegmentAnnotations(LitType):
  """Very general type for in-line text annotations, as List[AnnotationCluster].

  This is a more general version of SpanLabel, EdgeLabel, and other annotation
  types, designed to represent annotations that may span multiple segments.

  The basic unit is dtypes.AnnotationCluster, which contains a label, optional
  score, and one or more SpanLabel annotations, each of which points to a
  specific segment from the input.

  TODO(lit-dev): by default, spans are treated as bytes in this context.
  Make this configurable, if some spans need to refer to tokens instead.
  """
  exclusive: bool = False  # if true, treat as candidate list
  background: bool = False  # if true, don't emphasize in visualization


##
# Model internals, for interpretation.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Embeddings(LitType):
  """Embeddings or model activations, as fixed-length <float>[emb_dim]."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Gradients(LitType):
  """Gradients with respect to embeddings."""
  grad_for: Optional[Text] = None  # name of Embeddings field
  # Name of the field in the input that can be used to specify the target class
  # for the gradients.
  grad_target_field_key: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenEmbeddings(LitType):
  """Per-token embeddings, as <float>[num_tokens, emb_dim]."""
  align: Optional[Text] = None  # name of a Tokens field


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenGradients(LitType):
  """Gradients with respect to per-token inputs, as <float>[num_tokens, emb_dim]."""
  align: Optional[Text] = None  # name of a Tokens field
  grad_for: Optional[Text] = None  # name of TokenEmbeddings field
  # Name of the field in the input that can be used to specify the target class
  # for the gradients.
  grad_target_field_key: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageGradients(LitType):
  """Gradients with respect to per-pixel inputs, as a multidimensional array."""
  # Name of the field in the input for which the gradients are computed.
  align: Optional[Text] = None
  # Name of the field in the input that can be used to specify the target class
  # for the gradients.
  grad_target_field_key: Optional[Text] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class AttentionHeads(LitType):
  """One or more attention heads, as <float>[num_heads, num_tokens, num_tokens]."""
  # input and output Tokens fields; for self-attention these can be the same
  align_in: Text
  align_out: Text


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SubwordOffsets(LitType):
  """Offsets to align input tokens to wordpieces or characters, as List[int].

  offsets[i] should be the index of the first wordpiece for input token i.
  """
  align_in: Text  # name of field in data spec
  align_out: Text  # name of field in model output spec


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SparseMultilabel(LitType):
  """Sparse multi-label represented as a list of strings, as List[str]."""
  vocab: Optional[Sequence[Text]] = None  # label names
  default: Sequence[Text] = []
  # TODO(b/162269499) Migrate non-comma separators to custom type.
  separator: Text = ","  # Used for display purposes.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SparseMultilabelPreds(LitType):
  """Sparse multi-label predictions represented as a list of tuples.

  The tuples are of the label and the score. So as a List[(str, float)].
  """
  vocab: Optional[Sequence[Text]] = None  # label names
  parent: Optional[Text] = None
  default: Sequence[Text] = []


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class FieldMatcher(LitType):
  """For matching spec fields.

  The front-end will perform spec matching and fill in the vocab field
  accordingly. UI will materialize this to a dropdown-list.
  Use MultiFieldMatcher when your intent is selecting more than one field in UI.
  """
  spec: Text  # which spec to check, 'dataset', 'input', or 'output'.
  types: Union[Text, Sequence[Text]]  # types of LitType to match in the spec.
  vocab: Optional[Sequence[Text]] = None  # names matched from the spec.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MultiFieldMatcher(LitType):
  """For matching spec fields.

  The front-end will perform spec matching and fill in the vocab field
  accordingly. UI will materialize this to multiple checkboxes. Use this when
  the user needs to pick more than one field in UI.
  """
  spec: Text  # which spec to check, 'dataset', 'input', or 'output'.
  types: Union[Text, Sequence[Text]]  # types of LitType to match in the spec.
  vocab: Optional[Sequence[Text]] = None  # names matched from the spec.
  default: Sequence[Text] = []  # default names of selected items.
  select_all: bool = False  # Select all by default (overriddes default).


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenSalience(LitType):
  """Metadata about a returned token salience map, returned as dtypes.TokenSalience."""
  autorun: bool = False  # If the saliency technique is automatically run.
  signed: bool  # If the returned values are signed.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class FeatureSalience(LitType):
  """Metadata about a returned feature salience map, returned as dtypes.FeatureSalience."""
  autorun: bool = True  # If the saliency technique is automatically run.
  signed: bool  # If the returned values are signed.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageSalience(LitType):
  """Metadata about a returned image saliency.

  The data is returned as an image in the base64 URL encoded format, e.g.,
  data:image/jpg;base64,w4J3k1Bfa...
  """
  autorun: bool = False  # If the saliency technique is automatically run.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SequenceSalience(LitType):
  """Metadata about a returned sequence salience map, returned as dtypes.SequenceSalienceMap."""
  autorun: bool = False  # If the saliency technique is automatically run.
  signed: bool  # If the returned values are signed.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Boolean(LitType):
  """Boolean value."""
  default: bool = False


# LINT.ThenChange(../client/lib/types.ts)
