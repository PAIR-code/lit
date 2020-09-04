# Lint as: python3
"""Wrapper for HuggingFace models in LIT.

Includes BERT masked LM, GPT-2, and T5.

This wrapper loads a model into memory and implements the a number of helper
functions to predict a batch of examples and extract information such as
hidden states and attention.
"""
from typing import Dict, List, Tuple

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import numpy as np
import tensorflow as tf
from transformers import GPT2LMHeadModel
import torch
import transformers

from lit_nlp.examples.models.encoder import get_encoder

class BertMLM(lit_model.Model):
  """BERT masked LM using Huggingface Transformers and TensorFlow 2."""

  MASK_TOKEN = "[MASK]"

  @property
  def num_layers(self):
    return self.model.config.num_hidden_layers

  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, model_name="bert-base-uncased", use_tf=False, top_k=10):
    super().__init__()
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    self.use_tf = use_tf
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    if self.use_tf:
      self.model = transformers.TFBertForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True)
    else:
      self.model = transformers.BertForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True)
    self.top_k = top_k

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

    # slice attention, omitting [CLS] and [SEP]
    for i in range(len(range(self.num_layers))):
      output[f"layer_{i:d}_attention"] = output[f"layer_{i:d}_attention"][:, slicer, slicer]

    probas = output.pop("probas")

    # Predictions at every position, regardless of masking.
    output["pred_tokens"] = self._get_topk_tokens(probas[slicer])
    # Trim down to only the mask positions, to avoid sending a huge amount
    # of data.
    for i, token in enumerate(output["tokens"]):
      if token != self.MASK_TOKEN:
        output["pred_tokens"][i] = []

    return output

  ##
  # LIT API implementations
  def max_minibatch_size(self, unused_config=None) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs, config=None):
    """Predict on a single minibatch of examples."""
    # If input has a 'tokens' field, use that. Otherwise tokenize the text.
    tokenized_texts = [
        ex.get("tokens") or self.tokenizer.tokenize(ex["text"]) for ex in inputs
    ]
    # Process to ids, add special tokens, and compute segment ids and masks.

    if self.use_tf:
      encoded_input = self.tokenizer.batch_encode_plus(
            tokenized_texts,
            is_pretokenized=True,
            return_tensors="tf",
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True)
    else:
      encoded_input = self.tokenizer.batch_encode_plus(
            tokenized_texts,
            is_pretokenized=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True)
    # We have to set max_length explicitly above so that
    # max_tokens <= model_max_length, in order to avoid indexing errors. But
    # the combination of max_length=<integer> and pad_to_max_length=True means
    # that if the max is < model_max_length, we end up with extra padding.
    # Thee lines below strip this off.
    # TODO(lit-dev): submit a PR to make this possible with tokenizer options?
    if self.use_tf:
      max_tokens = tf.reduce_max(
            tf.reduce_sum(encoded_input["attention_mask"], axis=1))
    else:
      max_tokens = torch.max(
            torch.sum(encoded_input["attention_mask"], dim=1))
    encoded_input = {k: v[:, :max_tokens] for k, v in encoded_input.items()}


    # logits is a single tensor
    #    <float32>[batch_size, num_tokens, vocab_size]
    # embs is a list of num_layers + 1 tensors, each
    #    <float32>[batch_size, num_tokens, h_dim]
    # attentions is a list of num_layers tensors, each
    #    <float32>[batch_size, num_heads, num_tokens, num_tokens]

    if self.use_tf:
      logits, embs, unused_attentions = self.model(encoded_input)
    else:
      with torch.no_grad():
            logits, embs, unused_attentions = self.model(**encoded_input)

    if self.use_tf:
      batched_outputs = {
            "probas": tf.nn.softmax(logits, axis=-1).numpy(),
            "input_ids": encoded_input["input_ids"].numpy(),
            "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1).numpy(),
            "cls_emb": embs[-1][:, 0].numpy(),  # last layer, first token
        }
    else:
      batched_outputs = {
            "probas": torch.softmax(logits, dim=-1).numpy(),
            "input_ids": encoded_input["input_ids"].numpy(),
            "ntok": torch.sum(encoded_input["attention_mask"], dim=1).numpy(),
            "cls_emb": embs[-1][:, 0].numpy(),  # last layer, first token
        }
    # List of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(batched_outputs)

    # add attention to output.
    for i in range(len(unused_attentions)):
      batched_outputs[f"layer_{i:d}_attention"] = unused_attentions[i].numpy()

    # Postprocess to remove padding and decode predictions.
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return {
        "text": lit_types.TextSegment(),
        "tokens": lit_types.Tokens(required=False),
    }

  def output_spec(self):
    spec = {
          "tokens": lit_types.Tokens(parent="text"),
          "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
          "cls_emb": lit_types.Embeddings(),
      }
    for i in range(self.num_layers):
        spec[f"layer_{i:d}_attention"] = lit_types.AttentionHeads(
              align=("tokens", "tokens"))
    return spec


class GPT2LanguageModel(lit_model.Model):
  """Wrapper for a Huggingface Transformers GPT-2 model.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions plus additional helper functions to
  convert and clean tokens and to compute the top_k predictions from logits.
  """

  @property
  def num_layers(self):
    return self.model.config.n_layer

  def __init__(self, model_name="gpt2", use_tf=False, top_k=10):
    """Constructor for GPT2LanguageModel.

    Args:
      model_name: Specify the GPT-2 size [distil, small, medium, large, xl].
      top_k: How many predictions to prune.
    """
    super().__init__()
    # GPT2 is trained without pad_token, so pick arbitrary one and mask out.
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, pad_token="<pad>")

    self.use_tf = use_tf

    if self.use_tf:
      self.model = transformers.TFGPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True)
    else:
      self.model = transformers.GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True)
    self.top_k = top_k

  @staticmethod
  def clean_bpe_token(tok):
    if not tok.startswith("Ġ"):
      return "_" + tok
    else:
      return tok.replace("Ġ", "")

  def _detokenize(self, ids):
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_bpe_token(t) for t in tokens]

  def _pred(self, encoded_inputs):
    """Predicts one batch of tokenized text.

    Also performs some batch-level post-processing in TF.
    Single-example postprocessing is done in _postprocess(), and operates on
    numpy arrays.

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
      encoded_inputs: output of self.tokenizer.batch_encode_plus()

    Returns:
      payload: Dictionary with items described above, each as single Tensor.
    """
    with torch.no_grad():
      logits, _, states, attentions = self.model(encoded_inputs["input_ids"])

    if self.use_tf:
      model_probs = tf.nn.softmax(logits, axis=-1)
      top_k = tf.math.top_k(model_probs, k=self.top_k, sorted=True, name=None)
      batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }
    else:
      model_probs = torch.softmax(logits, dim=-1)
      value, indices = torch.topk(model_probs, k=self.top_k, sorted=True)
      batched_outputs = {
            "input_ids": encoded_inputs["input_ids"],
            "ntok": torch.sum(encoded_inputs["attention_mask"], dim=1),
            "top_k_indices": indices,
            "top_k_probs": value,
        }

    # Convert representations for each layer from tuples to single Tensor.
    for i in range(len(attentions)):
      batched_outputs[f"layer_{i:d}_attention"] = attentions[i]
    for i in range(len(states)):
      if self.use_tf:
        batched_outputs[f"layer_{i:d}_avg_embedding"] = tf.math.reduce_mean(
              states[i], axis=1)
      else:
        batched_outputs[f"layer_{i:d}_avg_embedding"] = torch.mean(
              states[i], dim=1)

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    ntok = preds.pop("ntok")
    ids = preds.pop("input_ids")[:ntok]
    preds["tokens"] = self._detokenize(ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a List[List[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop("top_k_probs")[:ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self._detokenize(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    return preds

  ##
  # LIT API implementations
  def max_minibatch_size(self, unused_config=None) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs, config=None):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [ex["text"] for ex in inputs]
    if self.use_tf:
        encoded_inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="tf",
            add_special_tokens=True,
            add_prefix_space=True,
            pad_to_max_length=True)
    else:
        encoded_inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            add_special_tokens=True,
            add_prefix_space=True,
            pad_to_max_length=True)
    # Get the predictions.
    batched_outputs = self._pred(encoded_inputs)
    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return {"text": lit_types.TextSegment()}

  def output_spec(self):
    spec = {
        # the "parent" keyword tells LIT which field in the input spec we should
        # compare this to when computing metrics.
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "tokens": lit_types.Tokens(parent="text"),  # all tokens
    }
    # Add attention and embeddings from each layer.
    for i in range(self.num_layers):
      spec[f"layer_{i:d}_attention"] = lit_types.AttentionHeads(
          align=("tokens", "tokens"))
      spec[f"layer_{i:d}_avg_embedding"] = lit_types.Embeddings()
    return spec




