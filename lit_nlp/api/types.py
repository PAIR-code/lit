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
import enum
import math
import numbers
from typing import Any, NewType, Optional, Sequence, Type, TypedDict, Union

import attr
from lit_nlp.api import dtypes
import numpy as np

JsonDict = dict[str, Any]
Input = NewType("Input", JsonDict)
ExampleId = NewType("ExampleId", str)
ScoredTextCandidates = Sequence[tuple[str, Optional[float]]]
TokenTopKPredsList = Sequence[ScoredTextCandidates]
NumericTypes = numbers.Number


class InputMetadata(TypedDict):
  added: Optional[bool]
  # pylint: disable=invalid-name
  parentId: Optional[ExampleId]   # Named to match TypeScript data structure
  # pylint: enable=invalid-name
  source: Optional[str]


class IndexedInput(TypedDict):
  data: Input
  id: ExampleId
  meta: InputMetadata


##
# Base classes, for common functionality and type grouping.
@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class LitType(metaclass=abc.ABCMeta):
  """Base class for LIT Types."""
  required: bool = True  # for input fields, mark if required by the model.
  annotated: bool = False  # If this type is created from an Annotator.
  show_in_data_table = True  # If true, show this info the data table.
  # TODO(lit-dev): Add defaults for all LitTypes
  default = None  # an optional default value for a given type.

  def validate_input(self, value: Any, spec: "Spec", example: Input):
    """Validate a dataset example's value against its spec in an example.

    Subtypes should override to validate a provided value and raise a ValueError
    if the value is not valid.

    Args:
      value: The value to validate against the specific LitType.
      spec: The spec of the dataset.
      example: The entire example of which the value is a part of.

    Raises:
      ValueError if validation fails.
    """
    pass

  def validate_output(self, value: Any, output_spec: "Spec",
                      output_dict: JsonDict, input_spec: "Spec",
                      dataset_spec: "Spec", input_example: Input):
    """Validate a model output value against its spec and input example.

    Subtypes should override to validate a provided value and raise a ValueError
    if the value is not valid.

    Args:
      value: The value to validate against the specific LitType.
      output_spec: The output spec of the model.
      output_dict: The entire model output for the example.
      input_spec: The input spec of the model.
      dataset_spec: The dataset spec.
      input_example: The example from which the output value is returned.

    Raises:
      ValueError if validation fails.
    """
    del output_spec, output_dict, dataset_spec
    # If not overwritten by a LitType, then validate it as an input to re-use
    # simple validation code.
    self.validate_input(value, input_spec, input_example)

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
    d["__name__"] = self.__class__.__name__
    return d

  @staticmethod
  def from_json(d: JsonDict):
    """Used by serialize.py.

    Args:
      d: The JSON Object-like dictionary to attempt to parse.

    Returns:
      An instance of a LitType subclass defined by the contents of `d`.

    Raises:
      KeyError: If `d` does not have a `__name__` property.
      NameError: If `d["__name__"]` is not a `LitType` subclass.
      TypeError: If `d["__name__"]` is not a string.
    """
    try:
      type_name = d.pop("__name__")
    except KeyError as e:
      raise KeyError("A __name__ property is required to parse a LitType from "
                     "JSON.") from e

    if not isinstance(type_name, str):
      raise TypeError("The value of __name__ must be a string.")

    base_cls = globals().get("LitType")
    cls = globals().get(type_name)  # class by name from this module
    if cls is None or not issubclass(cls, base_cls):
      raise NameError(f"{type_name} is not a valid LitType.")

    return cls(**d)

Spec = dict[str, LitType]

# Attributes that should be treated as a reference to other fields.
FIELD_REF_ATTRIBUTES = frozenset(
    {"parent", "align", "align_in", "align_out", "grad_for"})


def _remap_leaf(leaf: LitType, keymap: dict[str, str]) -> LitType:
  """Remap any field references on a LitType."""
  d = attr.asdict(leaf)  # mutable
  d = {
      k: (keymap.get(v, v) if k in FIELD_REF_ATTRIBUTES else v)
      for k, v in d.items()
  }
  return leaf.__class__(**d)


def remap_spec(spec: Spec, keymap: dict[str, str]) -> Spec:
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
class StringLitType(LitType):
  """User-editable text input.

  All automated edits are disabled for this type.

  Mainly used for string inputs that have special formatting, and should only
  be edited manually.
  """
  default: str = ""

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, str):
      raise ValueError(f"{value} is of type {type(value)}, expected str")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TextSegment(StringLitType):
  """Text input (untokenized), a single string."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageBytes(LitType):
  """An image, an encoded base64 ascii string (starts with 'data:image...')."""

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, str) or not value.startswith("data:image"):
      raise ValueError(f"{value} is not an encoded image string.")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class JPEGBytes(ImageBytes):
  """As ImageBytes, but assumed to be in jpeg format."""

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, str) or not value.startswith("data:image/jpg"):
      raise ValueError(f"{value} is not an encoded JPEG image string.")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class PNGBytes(ImageBytes):
  """As ImageBytes, but assumed to be in png format."""

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, str) or not value.startswith("data:image/png"):
      raise ValueError(f"{value} is not an encoded PNG image string.")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedText(TextSegment):
  """Generated (untokenized) text."""
  # Name of a TextSegment field to evaluate against
  parent: Optional[str] = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, str):
      raise ValueError(f"{value} is of type {type(value)}, expected str")
    if self.parent and not isinstance(input_spec[self.parent], TextSegment):
      raise ValueError(f"parent field {self.parent} is of type "
                       f"{type(self.parent)}, expected TextSegment")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ListLitType(LitType):
  """List type."""
  default: Sequence[Any] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class _StringCandidateList(ListLitType):
  """A list of (text, score) tuples."""
  default: ScoredTextCandidates = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, list):
      raise ValueError(f"{value} is not a list")

    for v in value:
      if not (isinstance(v, tuple) and isinstance(v[0], str) and
              (v[1] is None or isinstance(v[1], NumericTypes))):
        raise ValueError(f"{v} list item is not a (str, float) tuple)")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedTextCandidates(_StringCandidateList):
  """Multiple candidates for GeneratedText."""
  # Name of a TextSegment field to evaluate against
  parent: Optional[str] = None

  @staticmethod
  def top_text(value: ScoredTextCandidates) -> str:
    return value[0][0] if len(value) else ""

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    if self.parent and not isinstance(input_spec[self.parent], TextSegment):
      raise ValueError(f"parent field {self.parent} is of type "
                       f"{type(input_spec[self.parent])}, expected TextSegment")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ReferenceTexts(_StringCandidateList):
  """Multiple candidates for TextSegment."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TopTokens(_StringCandidateList):
  """Multiple tokens with weight."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class URLLitType(TextSegment):
  """TextSegment that should be interpreted as a URL."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedURL(TextSegment):
  """A URL that was generated as part of a model prediction."""
  align: Optional[str] = None  # name of a field in the model output

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(value, output_spec, output_dict, input_spec,
                            dataset_spec, input_example)
    if self.align and self.align not in output_spec:
      raise ValueError(f"aligned field {self.align} is not in output_spec")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SearchQuery(TextSegment):
  """TextSegment that should be interpreted as a search query."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class _StringList(ListLitType):
  """A list of strings."""
  default: Sequence[str] = []

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, list) or not all(
        [isinstance(v, str) for v in value]):
      raise ValueError(f"{value} is not a list of strings")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Tokens(_StringList):
  """Tokenized text."""
  default: Sequence[str] = attr.Factory(list)
  # Name of a TextSegment field from the input
  # TODO(b/167617375): should we use 'align' here?
  parent: Optional[str] = None
  mask_token: Optional[str] = None  # optional mask token for input
  token_prefix: Optional[str] = "##"  # optional prefix used in tokens


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenTopKPreds(ListLitType):
  """Predicted tokens, as from a language model.

  The inner list should contain (word, probability) in descending order.
  """
  default: Sequence[ScoredTextCandidates] = None

  align: str = None  # name of a Tokens field in the model output
  parent: Optional[str] = None

  def _validate_scored_candidates(self, scored_candidates):
    """Validates a list of scored candidates."""
    prev_val = math.inf
    for scored_candidate in scored_candidates:
      if not isinstance(scored_candidate, tuple):
        raise ValueError(f"{scored_candidate} is not a tuple")
      if not isinstance(scored_candidate[0], str):
        raise ValueError(f"{scored_candidate} first element is not a str")
      if scored_candidate[1] is not None:
        if not isinstance(scored_candidate[1], NumericTypes):
          raise ValueError(f"{scored_candidate} second element is not a num")
        if prev_val < scored_candidate[1]:
          raise ValueError(
              "TokenTopKPreds candidates are not in descending order")
        else:
          prev_val = scored_candidate[1]

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):

    if not isinstance(value, list):
      raise ValueError(f"{value} is not a list of scored text candidates")
    for scored_candidates in value:
      self._validate_scored_candidates(scored_candidates)
    if self.align and not isinstance(output_spec[self.align], Tokens):
      raise ValueError(
          f"aligned field {self.align} is {type(output_spec[self.align])}, "
          "expected Tokens")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Scalar(LitType):
  """Scalar value, a single float or int."""
  min_val: float = 0
  max_val: float = 1
  default: float = 0
  step: float = .01

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, NumericTypes):
      raise ValueError(f"{value} is of type {type(value)}, expected a number")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class RegressionScore(Scalar):
  """Regression score, a single float."""
  # name of a Scalar or RegressionScore field in input
  parent: Optional[str] = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, NumericTypes):
      raise ValueError(f"{value} is of type {type(value)}, expected a number")
    if self.parent and not isinstance(dataset_spec[self.parent], Scalar):
      raise ValueError(f"parent field {self.parent} is of type "
                       f"{type(self.parent)}, expected Scalar")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ReferenceScores(ListLitType):
  """Score of one or more target sequences."""
  default: Sequence[float] = None

  # name of a TextSegment or ReferenceTexts field in the input
  parent: Optional[str] = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if isinstance(value, list):
      if not all([isinstance(v, NumericTypes) for v in value]):
        raise ValueError(f"{value} is of type {type(value)}, expected a list "
                         "of numbers")
    elif not isinstance(value, np.ndarray) or not np.issubdtype(
        value.dtype, np.number):
      raise ValueError(f"{value} is of type {type(value)}, expected a list of "
                       "numbers")
    if self.parent and not isinstance(
        input_spec[self.parent], (TextSegment, ReferenceTexts)):
      raise ValueError(f"parent field {self.parent} is of type "
                       f"{type(self.parent)}, expected TextSegment or "
                       "ReferenceTexts")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CategoryLabel(StringLitType):
  """Category or class label, a single string."""
  # Optional vocabulary to specify allowed values.
  # If omitted, any value is accepted.
  vocab: Optional[Sequence[str]] = None  # label names

  def validate_input(self, value, spec: Spec, example: Input):
    if not isinstance(value, str):
      raise ValueError(f"{value} is of type {type(value)}, expected str")
    if self.vocab and value not in list(self.vocab):
      raise ValueError(f"{value} is not in provided vocab")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class _Tensor(LitType):
  """A tensor type."""
  default: Sequence[float] = None

  def validate_input(self, value, spec: Spec, example: Input):
    if isinstance(value, list):
      if not all([isinstance(v, NumericTypes) for v in value]):
        raise ValueError(f"{value} is not a list of numbers")
    elif isinstance(value, np.ndarray):
      if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{value} is not an array of numbers")
    else:
      raise ValueError(f"{value} is not a list or ndarray of numbers")

  def validate_ndim(self, value, ndim: Union[int, list[int]]):
    """Validate the number of dimensions in a tensor.

    Args:
      value: The tensor to validate.
      ndim: Either a number of dimensions to validate that the value has, or
        a list of dimensions any of which are valid for the value to have.

    Raises:
      ValueError if validation fails.
    """
    if isinstance(ndim, int):
      ndim = [ndim]
    if isinstance(value, np.ndarray):
      if value.ndim not in ndim:
        raise ValueError(f"{value} ndim is not one of {ndim}")
    else:
      if 1 not in ndim:
        raise ValueError(f"{value} ndim is not 1. "
                         "Use a numpy array for multidimensional arrays")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MulticlassPreds(_Tensor):
  """Multiclass predicted probabilities, as <float>[num_labels]."""
  # Vocabulary is required here for decoding model output.
  # Usually this will match the vocabulary in the corresponding label field.
  vocab: Sequence[str]  # label names
  null_idx: Optional[int] = None  # vocab index of negative (null) label
  parent: Optional[str] = None  # CategoryLabel field in input
  autosort: Optional[bool] = False  # Enable automatic sorting
  threshold: Optional[float] = None  # binary threshold, used to compute margin

  @property
  def num_labels(self):
    return len(self.vocab)

  def validate_input(self, value, spec: Spec, example: Input):
    super().validate_input(value, spec, example)
    if self.null_idx is not None:
      if self.null_idx < 0 or self.null_idx >= self.num_labels:
        raise ValueError(f"null_idx {self.null_idx} is not in the vocab range")

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    self.validate_input(value, output_spec, input_example)
    if self.parent and not isinstance(
        dataset_spec[self.parent], CategoryLabel):
      raise ValueError(f"parent field {self.parent} is of type "
                       f"{type(self.parent)}, expected CategoryLabel")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SequenceTags(_StringList):
  """Sequence tags, aligned to tokens.

  The data should be a list of string labels, one for each token.
  """
  align: str  # name of Tokens field


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SpanLabels(ListLitType):
  """Span labels aligned to tokens.

  Span labels can cover more than one token, may not cover all tokens in the
  sentence, and may overlap with each other.
  """
  default: Sequence[dtypes.SpanLabel] = None
  align: str  # name of Tokens field
  parent: Optional[str] = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, list) or not all(
        [isinstance(v, dtypes.SpanLabel) for v in value]):
      raise ValueError(f"{value} is not a list of SpanLabels")
    if not isinstance(output_spec[self.align], Tokens):
      raise ValueError(f"{self.align} is not a Tokens field")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class EdgeLabels(ListLitType):
  """Edge labels between pairs of spans.

  This is a general form for structured prediction output; each entry consists
  of (span1, span2, label). See
  https://arxiv.org/abs/1905.06316 (Tenney et al. 2019) and
  https://github.com/nyu-mll/jiant/tree/master/probing#data-format for more
  details.
  """
  default: Sequence[dtypes.EdgeLabel] = None
  align: str  # name of Tokens field

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, list) or not all(
        [isinstance(v, dtypes.EdgeLabel) for v in value]):
      raise ValueError(f"{value} is not a list of EdgeLabel")
    if not isinstance(output_spec[self.align], Tokens):
      raise ValueError(f"{self.align} is not a Tokens field")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MultiSegmentAnnotations(ListLitType):
  """Very general type for in-line text annotations.

  This is a more general version of SpanLabel, EdgeLabel, and other annotation
  types, designed to represent annotations that may span multiple segments.

  The basic unit is dtypes.AnnotationCluster, which contains a label, optional
  score, and one or more SpanLabel annotations, each of which points to a
  specific segment from the input.

  TODO(lit-dev): by default, spans are treated as bytes in this context.
  Make this configurable, if some spans need to refer to tokens instead.
  """
  default: Sequence[dtypes.AnnotationCluster] = None
  exclusive: bool = False  # if true, treat as candidate list
  background: bool = False  # if true, don't emphasize in visualization

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    if not isinstance(value, list) or not all(
        [isinstance(v, dtypes.AnnotationCluster) for v in value]):
      raise ValueError(f"{value} is not a list of AnnotationCluster")

##
# Model internals, for interpretation.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Embeddings(_Tensor):
  """Embeddings or model activations, as fixed-length <float>[emb_dim]."""

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(value, output_spec, output_dict, input_spec,
                            dataset_spec, input_example)
    self.validate_ndim(value, 1)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class _GradientsBase(_Tensor):
  """Shared gradient attributes."""
  align: Optional[str] = None  # name of a Tokens field
  grad_for: Optional[str] = None  # name of Embeddings field
  # Name of the field in the input that can be used to specify the target class
  # for the gradients.
  grad_target_field_key: Optional[str] = None

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    if self.align is not None:
      align_entry = (output_spec[self.align] if self.align in output_spec
                     else input_spec[self.align])
      if not isinstance(align_entry, (Tokens, ImageBytes)):
        raise ValueError(f"{self.align} is not a Tokens or ImageBytes field")
    if self.grad_for is not None and not isinstance(
        output_spec[self.grad_for], (Embeddings, TokenEmbeddings)):
      raise ValueError(f"{self.grad_for} is not a Embeddings field")
    if (self.grad_target_field_key is not None and
        self.grad_target_field_key not in input_spec):
      raise ValueError(f"{self.grad_target_field_key} is not in input_spec")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Gradients(_GradientsBase):
  """1D gradients with respect to embeddings."""

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, 1)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class _InfluenceEncodings(_Tensor):
  """A single vector of <float>[enc_dim]."""
  grad_target: Optional[str] = None  # class for computing gradients (string)

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, 1)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenEmbeddings(_Tensor):
  """Per-token embeddings, as <float>[num_tokens, emb_dim]."""
  align: Optional[str] = None  # name of a Tokens field

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, 2)
    if self.align is not None and not isinstance(
        output_spec[self.align], Tokens):
      raise ValueError(f"{self.align} is not a Tokens field")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenGradients(_GradientsBase):
  """Gradients for per-token inputs, as <float>[num_tokens, emb_dim]."""

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, 2)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageGradients(_GradientsBase):
  """Gradients with respect to per-pixel inputs, as a multidimensional array."""

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, [2, 3])


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class AttentionHeads(_Tensor):
  """One or more attention heads, as <float>[num_heads, num_tokens, num_tokens]."""
  # input and output Tokens fields; for self-attention these can be the same
  align_in: str
  align_out: str

  def validate_output(self, value, output_spec: Spec, output_dict: JsonDict,
                      input_spec: Spec, dataset_spec: Spec,
                      input_example: Input):
    super().validate_output(
        value, output_spec, output_dict, input_spec, dataset_spec,
        input_example)
    self.validate_ndim(value, 3)
    if self.align_in is None or not isinstance(
        output_spec[self.align_in], Tokens):
      raise ValueError(f"{self.align_in} is not a Tokens field")
    if self.align_out is None or not isinstance(
        output_spec[self.align_out], Tokens):
      raise ValueError(f"{self.align_out} is not a Tokens field")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SubwordOffsets(ListLitType):
  """Offsets to align input tokens to wordpieces or characters.

  offsets[i] should be the index of the first wordpiece for input token i.
  """
  default: Sequence[int] = None
  align_in: str  # name of field in data spec
  align_out: str  # name of field in model output spec


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SparseMultilabel(_StringList):
  """Sparse multi-label represented as a list of strings."""
  vocab: Optional[Sequence[str]] = None  # label names
  separator: str = ","  # Used for display purposes.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SparseMultilabelPreds(_StringCandidateList):
  """Sparse multi-label predictions represented as a list of tuples.

  The tuples are of the label and the score.
  """
  default: ScoredTextCandidates = None
  vocab: Optional[Sequence[str]] = None  # label names
  parent: Optional[str] = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class FieldMatcher(LitType):
  """For matching spec fields.

  The front-end will perform spec matching and fill in the vocab field
  accordingly.
  """
  spec: str  # which spec to check, 'dataset', 'input', or 'output'.
  types: Union[str, Sequence[str]]  # types of LitType to match in the spec.
  vocab: Optional[Sequence[str]] = None  # names matched from the spec.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SingleFieldMatcher(FieldMatcher):
  """For matching a single spec field.

  UI will materialize this to a dropdown-list.
  """
  default: str = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MultiFieldMatcher(FieldMatcher):
  """For matching multiple spec fields.

  UI will materialize this to multiple checkboxes. Use this when the user needs
  to pick more than one field in UI.
  """
  default: Sequence[str] = []  # default names of selected items.
  select_all: bool = False  # Select all by default (overriddes default).


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Salience(LitType):
  """Metadata about a returned salience map."""
  autorun: bool = False  # If the saliency technique is automatically run.
  signed: bool  # If the returned values are signed.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenSalience(Salience):
  """Metadata about a returned token salience map."""
  default: dtypes.TokenSalience = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class FeatureSalience(Salience):
  """Metadata about a returned feature salience map."""
  default: dtypes.FeatureSalience = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ImageSalience(Salience):
  """Metadata about a returned image saliency.

  The data is returned as an image in the base64 URL encoded format, e.g.,
  data:image/jpg;base64,w4J3k1Bfa...
  """
  signed: bool = False  # If the returned values are signed.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SequenceSalience(Salience):
  """Metadata about a returned sequence salience map."""
  default: dtypes.SequenceSalienceMap = None


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class BooleanLitType(LitType):
  """Boolean value."""
  default: bool = False

  def validate_input(self, value, spec, example: Input):
    if not isinstance(value, bool):
      raise ValueError(f"{value} is not a boolean")


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CurveDataPoints(LitType):
  """Represents data points of a curve.

  A list of tuples where the first and second elements of the tuple are the
  x and y coordinates of the corresponding curve point respectively.
  """
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class InfluentialExamples(LitType):
  """Represents influential examples from the training set.

  This is as returned by a training-data attribution method like TracIn or
  influence functions.

  This describes a generator component; values are Sequence[Sequence[JsonDict]].
  """
  pass


@enum.unique
class MetricBestValue(dtypes.EnumSerializableAsValues, enum.Enum):
  """The method to use to determine the best value for a Metric."""
  HIGHEST = "highest"
  LOWEST = "lowest"
  NONE = "none"
  ZERO = "zero"


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MetricResult(LitType):
  """Score returned from the computation of a Metric."""
  default: float = 0
  description: str = ""
  best_value: MetricBestValue = MetricBestValue.NONE


# LINT.ThenChange(../client/lib/lit_types.ts)

# Type aliases for backend use.
# The following names are existing datatypes in TypeScript, so we add a
# `LitType` suffix to avoid collisions with language features on the front-end.
Boolean = BooleanLitType
String = StringLitType
URL = URLLitType


def get_type_by_name(typename: str) -> Type[LitType]:
  cls = globals()[typename]
  assert issubclass(cls, LitType)
  return cls
