"""Helpers to apply a subword tokenizer while retaining original token boundaries."""
import itertools
from typing import Sequence, Text, Callable

import numpy as np

# tokenizer: str -> list(str)
TokenizerFn = Callable[[Text], Sequence[Text]]


def flatten(lists):
  """Flatten a list-of-lists."""
  return list(itertools.chain.from_iterable(lists))


def subtokenize(tokens: Sequence[Text], subtokenizer_fn: TokenizerFn):
  """Apply a sub-word tokenizer and return start indices for the original token boundaries.

  The offsets returned by this can be used directly to project token indices to
  the new tokenization. For example:
    span = [i, j]  # end-exclusive,
    pieces, offsets = subtokenize(tokens, my_tokenizer_fn)
    wpm_span = offsets[span]  # re-map to subword indices

  Args:
    tokens: list of strings
    subtokenizer_fn: function that maps string -> list of strings

  Returns:
    pieces: (np.ndarray of string) subword strings
    offsets: (np.ndarray) indices into pieces corresponding to the start of each
    original token. len(offsets) = len(pieces) + 1, and offsets[-1] ==
    len(pieces).
  """
  # Subwords for each token, as list-of-lists
  splits = [subtokenizer_fn(token) for token in tokens]
  lengths = np.array([0] + [len(split) for split in splits])
  return flatten(splits), np.cumsum(lengths)
