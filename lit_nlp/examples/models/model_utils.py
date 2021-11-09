"""Helpers for working with transformers models."""
from typing import List, Optional

from absl import logging
import transformers


def load_pretrained(cls, *args, **kw):
  """Load a transformers model in TF2, with fallback to PyTorch weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from PyTorch checkpoint (from_pt=True)")
    return cls.from_pretrained(*args, from_pt=True, **kw)


def batch_encode_pretokenized(tokenizer: transformers.PreTrainedTokenizerBase,
                              tokenized_inputs: List[List[str]],
                              tokenized_pair_inputs: Optional[List[
                                  List[str]]] = None,
                              tensor_type="tf",
                              **kw) -> transformers.BatchEncoding:
  """Batch encode pre-tokenized text, without further splitting.

  This is necessary because tokenizer(..., is_split_into_words=True) doesn't
  guarantee that tokens will stay intact - only that the final tokens will not
  span the given boundaries. If the tokenizer is called directly, you'll get
  things like: "foo" "##bar" -> "foo" "#" "#" "bar"

  Based on the implementation of batch_encode_plus in
  https://github.com/huggingface/transformers/blob/v4.1.1/src/transformers/tokenization_utils_base.py#L2489

  Args:
    tokenizer: Transformers tokenizer
    tokenized_inputs: list of tokenized inputs
    tokenized_pair_inputs: (optional) list of tokenized second-segment inputs
    tensor_type: tensor type to return
    **kw: additional args, forwarded to tokenizer.prepare_for_model

  Returns:
    BatchEncoding, suitable for model input
  """
  encoded_input = {}
  tokenized_pair_inputs = (
      tokenized_pair_inputs or [None] * len(tokenized_inputs))
  for tokens, pair_tokens in zip(tokenized_inputs, tokenized_pair_inputs):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    pair_ids = (
        tokenizer.convert_tokens_to_ids(pair_tokens)
        if pair_tokens is not None else None)
    encoded = tokenizer.prepare_for_model(
        ids,
        pair_ids=pair_ids,
        add_special_tokens=True,
        padding="do_not_pad",
        truncation="longest_first",
        return_attention_mask=False,
        pad_to_multiple_of=False,
        **kw)
    for k, v in encoded.items():
      encoded_input.setdefault(k, []).append(v)

  encoded_input = tokenizer.pad(
      encoded_input, padding="longest", return_attention_mask=True)
  return transformers.BatchEncoding(encoded_input, tensor_type=tensor_type)
