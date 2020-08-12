# Lint as: python3
"""Wrapper for HuggingFace models in LIT.

Includes BERT masked LM, GPT-2, and T5.

This wrapper loads a model into memory and implements the a number of helper
functions to predict a batch of examples and extract information such as
hidden states and attention.
"""
from typing import Any, Dict, List, Text, Tuple

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import numpy as np
import tensorflow as tf
import transformers

MAX_SEQ_LENGTH = 512


class BertMLM(lit_model.Model):
  """BERT masked LM using Huggingface Transformers and TensorFlow 2."""

  MASK_TOKEN = "[MASK]"

  def __init__(self, model_name="bert-base-uncased", top_k=10):
    super().__init__()
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    self.model = transformers.TFBertForMaskedLM.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True)
    self.top_k = top_k

  def _batch_encode(self, text: List[str]):
    """Encode a batch of strings for model input."""
    return self.tokenizer.batch_encode_plus(
        text,
        return_tensors="tf",
        add_special_tokens=True,
        pad_to_max_length=True)

  # TODO(lit-dev): break this out as a helper function, write some tests,
  # and de-duplicate code with the other text generation functions.
  def _get_topk_tokens(self,
                       scores: np.ndarray) -> List[List[Tuple[str, float]]]:
    """Convert raw scores to top-k token predictions."""
    # scores is [num_tokens, vocab_size]
    # Find the vocab indices of top k predictions, at each token.
    # np.argpartition is faster than a full argsort for k << V,
    # but we need to sort the output after slicing (see below).
    index_array = np.argpartition(scores, -self.top_k, axis=1)[:, -self.top_k:]
    # These are each [num_tokens, tok_k]
    top_tokens = [
        self.tokenizer.convert_ids_to_tokens(idxs) for idxs in index_array
    ]
    top_scores = np.take_along_axis(scores, index_array, axis=1)
    # Convert to a list of lists of (token, score) pairs,
    # where inner lists are sorted in descending order of score.
    return [
        sorted(list(zip(toks, scores)), key=lambda ab: -ab[1])
        for toks, scores in zip(top_tokens, top_scores)
    ]
    # TODO(lit-dev): consider returning indices and a vocab, since repeating
    # strings is slow and redundant.

  def _postprocess(self, output: Dict[str, np.ndarray]):
    """Postprocess, modifying output dict in-place."""
    # Slice to remove padding, omitting initial [CLS] and final [SEP]
    slicer = slice(1, output.pop("ntok") - 1)
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(
        output.pop("input_ids")[slicer])
    probas = output.pop("probas")

    # Predictions at every position, regardless of masking.
    output["pred_tokens"] = self._get_topk_tokens(probas[slicer])
    # Trim down to only the mask positions, to avoid sending a huge amount
    # of data.
    for i, token in enumerate(output["tokens"]):
      if token != self.MASK_TOKEN:
        output["pred_tokens"][i] = []

    return output

  def _predict_minibatch(self, texts: List[str]):
    """Run the model on a batch of texts."""
    encoded_input = self._batch_encode(texts)
    # logits is a single tensor
    #    <float32>[batch_size, num_tokens, vocab_size]
    # embs is a list of num_layers + 1 tensors, each
    #    <float32>[batch_size, num_tokens, h_dim]
    # attentions is a list of num_layers tensors, each
    #    <float32>[batch_size, num_heads, num_tokens, num_tokens]
    logits, embs, unused_attentions = self.model(encoded_input)
    batched_outputs = {
        "probas": tf.nn.softmax(logits, axis=-1).numpy(),
        "input_ids": encoded_input["input_ids"].numpy(),
        "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1).numpy(),
        "cls_emb": embs[-1][:, 0].numpy(),  # last layer, first token
    }
    # List of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(batched_outputs)
    # Postprocess to remove padding and decode predictions.
    return map(self._postprocess, unbatched_outputs)

  ##
  # LIT API implementations
  def max_minibatch_size(self, unused_config=None) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs, config=None):
    """Predict on a single minibatch of examples."""
    return self._predict_minibatch([ex["text"] for ex in inputs])

  def input_spec(self):
    return {"text": lit_types.TextSegment()}

  def output_spec(self):
    return {
        "tokens": lit_types.Tokens(parent="text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "cls_emb": lit_types.Embeddings(),
    }


class GPT2LanguageModel(lit_model.Model):
  """Wrapper for a GPT-2 language model."""

  def __init__(self, *args, **kw):
    # This loads the checkpoint into memory, so we"re ready for interactive use.
    self._model = HFGPT2(*args, **kw)

  # LIT API implementations
  def max_minibatch_size(self, unused_config=None) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs, config=None):
    """Predict on a single minibatch of examples."""
    examples = [self._model.convert_dict_input(d) for d in inputs]
    payload = self._model.predict_examples(examples)
    return payload

  def input_spec(self):
    return {
        "text": lit_types.TextSegment(),
    }

  def output_spec(self):
    spec = {
        # the "parent" keyword tells LIT which field in the input spec we should
        # compare this to when computing metrics.
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "tokens": lit_types.Tokens(parent="text"),  # all tokens
    }
    # Add attention for each layer.
    for i in range(self._model.get_num_layers()):
      spec[f"layer_{i:d}_attention"] = lit_types.AttentionHeads(
          align=("tokens", "tokens"))
      spec[f"layer_{i:d}_avg_embedding"] = lit_types.Embeddings()
    return spec


class HFGPT2(object):
  """Wrapper for a Huggingface Transformers GPT-2 model.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions plus additional helper functions to
  convert and clean tokens and to compute the top_k predictions from logits.
  """

  def __init__(self, model_name="gpt2", top_k=10):
    """Constructor for HFGPT2 class.

    Args:
      model_name: Specify the GPT-2 size [distil, small, medium, large, xl].
      top_k: How many predictions to prune.
    """

    super().__init__()
    # GPT2 is trained without pad_token, so pick arbitrary one and mask out.
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, pad_token="<pad>")
    self.model = transformers.TFGPT2LMHeadModel.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True)
    self.top_k = top_k

  def _tokenize(self, text: List[str]):
    """Function to tokenize a batch of strings.

    Args:
      text: A list of strings to analyze.

    Returns:
      tok_input: Dictionary of input_ids, token_type_ids, and attention_mask;
                 Each is tf.Tensor of shape (batch_size, max_len_within_batch).
    """
    tok_input = self.tokenizer.batch_encode_plus(
        text,
        # Specify TF over PyTorch.
        return_tensors="tf",
        # For sequence boundaries.
        add_special_tokens=True,
        # Otherwise interpreted as starting with space.
        add_prefix_space=True,
        # Pad up to the max length inside the batch.
        pad_to_max_length=True)
    return tok_input

  def _clean_bpe(self, tokens):
    """Converts the special BPE tokens into readable format."""

    def clean_token(tok):
      if not tok.startswith("Ġ"):
        return "_" + tok
      else:
        return tok.replace("Ġ", "")

    if isinstance(tokens, list):
      return [clean_token(t) for t in tokens]
    else:
      return clean_token(tokens)

  def _detokenize(self, tokenized_text):
    """Convert back from tokenized dict to List[str]."""
    tokens = []
    for ids, mask in zip(tokenized_text["input_ids"],
                         tokenized_text["attention_mask"]):
      # Filter out padding and remove BPE continuation token.
      example = self._clean_bpe(
          self.tokenizer.convert_ids_to_tokens(
              [t for ix, t in enumerate(ids) if mask[ix] != 0]))
      tokens.append(example)
    return tokens

  def _pred(self, tokenized_text):
    """Predicts one batch of tokenized text.

    Each prediction has the following returns:
    logits: tf.Tensor (batch_size, sequence_length, config.vocab_size).
    past: List[tf.Tensor] of length config.n_layers with each tensor shape
             (2, batch_size, num_heads, sequence_length, embed_size_per_head)).
    states: Tuple of tf.Tensor (one for embeddings + one for each layer),
            with shape (batch_size, sequence_length, hidden_size).
    attentions: Tuple of tf.Tensor (one for each layer) with shape
                (batch_size, num_heads, sequence_length, sequence_length)
    Within this function, we combine each Tuple/List into a single Tensor.

    Args:
      tokenized_text: Dictionary with output from self._tokenize

    Returns:
      payload: Dictionary with items described above, each as single Tensor.
    """
    logits, _, states, attentions = self.model(tokenized_text["input_ids"])
    # Convert representations for each layer from tuples to single Tensor.
    payload = {}
    for i in range(len(attentions)):
      payload[f"layer_{i:d}_attention"] = attentions[i].numpy()
    for i in range(len(states)):
      payload[f"layer_{i:d}_avg_embedding"] = tf.math.reduce_mean(
          states[i], axis=1).numpy()

    payload["pred_tokens"] = self._logits_to_topk_probs(logits, tokenized_text)
    return payload

  def _logits_to_topk_probs(self, logits, tokenized_input):
    """Softmaxes the logits and prunes to the top k (token, prob) tuples."""
    model_probs = tf.nn.softmax(logits, axis=-1)
    top_k = tf.math.top_k(model_probs, k=self.top_k, sorted=True, name=None)
    indices = top_k.indices
    probs = top_k.values
    format_top_k = []
    for index_batch, prob_batch, mask_batch in zip(
        indices.numpy(), probs.numpy(),
        tokenized_input["attention_mask"].numpy()):
      # Initialize prediction for 0th token as N/A.
      formatted_batch = [[("N/A", 1.)]]
      # Add all other predictions for tokens.
      for index, prob, mask in zip(index_batch, prob_batch, mask_batch):
        if mask == 1:
          formatted_batch.append([(i, "{:.3f}".format(p)) for i, p in zip(
              self._clean_bpe(self.tokenizer.convert_ids_to_tokens(index)),
              prob)])
      format_top_k.append(formatted_batch)
    return format_top_k

  def _postproc_preds(self, pred):
    """Postprocessing done on each batch element."""
    return pred

  def convert_dict_input(self, input_dict: Dict[Text, Any]) -> Dict[Text, Any]:
    """Default implementation with generic keys."""
    return {
        "text": input_dict["text"],
        "guid": input_dict.get("guid", ""),
    }

  def get_num_layers(self):
    return self.model.config.n_layer

  def predict_examples(self, examples):
    """Public Function for LITModel to call on examples."""
    # Text as sequence of BPE ID"s.
    input_ids = self._tokenize([e["text"] for e in examples])
    # Get the predictions.
    preds = self._pred(input_ids)
    # detokenize BPE tokens for interface.
    detok = self._detokenize(input_ids)
    preds["tokens"] = detok

    payload = [self._postproc_preds(p) for p in utils.unbatch_preds(preds)]
    return payload

