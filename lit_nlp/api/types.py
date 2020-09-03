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
from typing import Dict, Text, Tuple, Sequence, Optional, Any

import attr

JsonDict = Dict[Text, Any]
ExampleId = Text


##
# Base classes, for common functionality and type grouping.
@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class LitType(metaclass=abc.ABCMeta):
  """Base class for LIT Types."""
  required: bool = True  # for input fields, mark if required by the model.

  def is_compatible(self, other):
    """Check equality, ignoring some fields."""
    if type(self) != type(other):  # pylint: disable=unidiomatic-typecheck
      return False
    d1 = attr.asdict(self)
    d1.pop('required', None)
    d2 = attr.asdict(other)
    d2.pop('required', None)
    return d1 == d2

Spec = Dict[Text, LitType]


##
# Concrete type clases


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TextSegment(LitType):
  """Text input (untokenized), a single string."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GeneratedText(TextSegment):
  """Generated (untokenized) text."""
  parent: Optional[Text] = None  # name of a TextSegment field, to compare to


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Tokens(LitType):
  """Tokenized text, as List[str]."""
  # TODO(lit-dev): should we use 'align' here?
  parent: Optional[Text] = None  # name of a TextSegment field in the input


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenTopKPreds(LitType):
  """Predicted tokens, as from a language model.

  Data should be a List[List[Tuple[str, float]]], where the inner list contains
  (word, probability) in descending order.
  """
  align: Text = None  # name of a Tokens field in the model output


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Scalar(LitType):
  """Scalar value, a single float."""
  # TODO(lit-dev): support optional range information, to use for legends,
  # plot bounds, etc. on frontend.
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class RegressionScore(Scalar):
  """Regression score, a single float."""
  # name of a Scalar or RegressionScore field in input
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
  parent: Optional[Text] = None  # name of CategoryLabel field in input

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

##
# Model internals, for interpretation.


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class Embeddings(LitType):
  """Embeddings or model activations, as fixed-length <float>[emb_dim]."""
  pass


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class TokenGradients(LitType):
  """Gradients with respect to inputs, as <float>[num_tokens, emb_dim]."""
  align: Optional[Text] = None  # path to tokens or other sequence field
  # TODO(lit-team): add pointer to an input embedding field, for implementing
  # things like path-integrated gradients?


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class AttentionHeads(LitType):
  """One or more attention heads, as <float>[num_heads, num_tokens, num_tokens]."""
  align: Tuple[Text, Text]  # paths to tokens fields
