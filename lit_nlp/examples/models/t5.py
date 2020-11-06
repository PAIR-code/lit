# Lint as: python3
"""Wrapper for HuggingFace implementation of T5."""
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils

import tensorflow as tf
import transformers

from rouge_score import rouge_scorer


def masked_token_mean(vectors, masks):
  """Mean over tokens.

  Args:
    vectors: <tf.float32>[batch_size, num_tokens, emb_dim]
    masks: <tf.int32>[batch_size, num_tokens]

  Returns:
    <tf.float32>[batch_size, emb_dim]
  """
  masks = tf.cast(masks, tf.float32)
  weights = masks / tf.reduce_sum(masks, axis=1, keepdims=True)
  return tf.reduce_sum(vectors * tf.expand_dims(weights, axis=-1), axis=1)


@attr.s(auto_attribs=True, kw_only=True)
class T5ModelConfig(object):
  """Config options for a T5 generation model."""
  # Preprocessing options
  inference_batch_size: int = 4
  # Input options
  input_prefix: str = ""
  # Output options
  max_gen_length: int = 50
  top_k: int = 10
  output_attention: bool = False


class T5GenerationModel(lit_model.Model):
  """Wrapper for a T5 model, implementing the LIT API."""

  @property
  def num_layers(self):
    return self.model.config.num_layers

  def __init__(self, model_name="t5-small", **config_kw):
    self.config = T5ModelConfig(**config_kw)
    self.tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    self.model = transformers.TFT5ForConditionalGeneration.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=self.config.output_attention)

    # TODO(gehrmann): temp solution for ROUGE.
    self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

  def _encode_texts(self, texts: List[str]):
    return self.tokenizer.batch_encode_plus(
        texts, return_tensors="tf", pad_to_max_length=True)

  def _force_decode(self, encoded_inputs, encoded_targets):
    """Get predictions for a batch of tokenized examples.

    Each forward pass produces the following:
      logits: batch_size x dec_len x vocab_size
      decoder_past_key_value_states: tuple with cached outputs.
      dec_states: tuple[len:dec_layers]:
                  batch_size x dec_len x hid_size
      dec_attn: [optional] tuple[len:dec_layers+1]
                batch_size x num_heads x dec_len x dec_len
      enc_final_state: batch_size x enc_len x hid_size
      enc_states: tuple[len:enc_layers]:
                  batch_size x enc_len x hid_size
      enc_attn: [optional] tuple[len:enc_layers+1]
                batch_size x num_heads x enc_len x enc_len

    The two optional attention fields are only returned if
    config.output_attention is set.

    Args:
      encoded_inputs: Dict as returned from Tokenizer for inputs.
      encoded_targets: Dict as returned from Tokenizer for outputs

    Returns:
      batched_outputs: Dict[str, tf.Tensor]
    """
    results = self.model(
        inputs=encoded_inputs["input_ids"],
        decoder_input_ids=encoded_targets["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        decoder_attention_mask=encoded_targets["attention_mask"],
        lm_label=encoded_targets["input_ids"])
    if self.config.output_attention:
      # Access the optional positional returns.
      dec_attn = results.pop(3)
      enc_attn = results.pop()
    logits, _, dec_states, enc_final_state, enc_states = results
    # While we are not using them, the deleted embeddings could be processed.
    del dec_states
    del enc_states

    model_probs = tf.nn.softmax(logits, axis=-1)
    top_k = tf.math.top_k(
        model_probs, k=self.config.top_k, sorted=True, name=None)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "input_ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
        "target_ids": encoded_targets["input_ids"],
        "target_ntok": tf.reduce_sum(encoded_targets["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }
    # enc_final_state is <float>[batch_size, num_tokens, emb_dim]
    # take the mean over real tokens to get <float>[batch_size, emb_dim]
    batched_outputs["encoder_final_embedding"] = masked_token_mean(
        enc_final_state, encoded_inputs["attention_mask"])

    if self.config.output_attention:
      for i in range(len(dec_attn)):
        batched_outputs[f"decoder_layer_{i:d}_attention"] = dec_attn[i]
      for i in range(len(enc_attn)):
        batched_outputs[f"encoder_layer_{i:d}_attention"] = enc_attn[i]

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Return tokenization for input text.
    input_ntok = preds.pop("input_ntok")
    input_ids = preds.pop("input_ids")[:input_ntok]
    preds["input_tokens"] = self.tokenizer.convert_ids_to_tokens(input_ids)
    # Return tokenization for target text.
    target_ntok = preds.pop("target_ntok")
    target_ids = preds.pop("target_ids")[:target_ntok]
    preds["target_tokens"] = self.tokenizer.convert_ids_to_tokens(target_ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a List[List[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:target_ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop(
        "top_k_probs")[:target_ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self.tokenizer.convert_ids_to_tokens(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    # Decode generated ids
    preds["generation"] = self.tokenizer.decode(
        preds.pop("generated_ids"), skip_special_tokens=True)

    # Process attention fields, if present.
    for key in preds:
      if not re.match(r"\w+_layer_(\d+)/attention", key):
        continue
      if key.startswith("encoder_"):
        ntok = input_ntok
      elif key.startswith("decoder_"):
        ntok = target_ntok
      else:
        raise ValueError(f"Invalid attention key: '{key}'")
      # Select only real tokens, since most of this matrix is padding.
      # <float32>[num_heads, max_seq_length, max_seq_length]
      # -> <float32>[num_heads, num_tokens, num_tokens]
      preds[key] = preds[key][:, :ntok, :ntok].transpose((0, 2, 1))
      # Make a copy of this array to avoid memory leaks, since NumPy otherwise
      # keeps a pointer around that prevents the source array from being GCed.
      preds[key] = preds[key].copy()

    return preds

  def _predict_minibatch_internal(self, inputs):
    """Run model on a single batch.

    Args:
      inputs: List[Dict] with fields as described by input_spec()

    Returns:
      outputs: List[Dict] with fields as described by output_spec()
    """
    # Text as sequence of sentencepiece ID"s.
    encoded_inputs = self._encode_texts([
        self.config.input_prefix + ex["input_text"] + " </s>" for ex in inputs
    ])
    encoded_targets = self._encode_texts(
        [ex.get("target_text", "") for ex in inputs])
    ##
    # Force-decode on target text, and also get encoder embs and attention.
    batched_outputs = self._force_decode(encoded_inputs, encoded_targets)
    # Get the conditional generation from the model.
    # Workaround for output_hidden not being compatible with generate.
    # See https://github.com/huggingface/transformers/issues/8361
    self.model.encoder.output_hidden_states = False
    self.model.decoder.output_hidden_states = False
    batched_outputs["generated_ids"] = self.model.generate(
        encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        max_length=self.config.max_gen_length)
    self.model.encoder.output_hidden_states = True
    self.model.decoder.output_hidden_states = True

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  ##
  # LIT API implementations
  def max_minibatch_size(self, unused_config=None) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 4

  def predict_minibatch(self, inputs, config=None):
    """Predict on a single minibatch of examples."""
    model_outputs = list(self._predict_minibatch_internal(inputs))

    # TODO(gehrmann): temp solution to get ROUGE scores in data table.
    for ex, mo in zip(inputs, model_outputs):
      score = self._scorer.score(
          target=ex["target_text"], prediction=mo["generation"])
      mo["rougeL"] = float(score["rougeL"].fmeasure)
    return model_outputs

  def input_spec(self):
    return {
        "input_text": lit_types.TextSegment(),
        # optional target text; if given will run force-decoding.
        "target_text": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    spec = {
        "input_tokens": lit_types.Tokens(parent="input_text"),
        "generation": lit_types.GeneratedText(parent="target_text"),
        "encoder_final_embedding": lit_types.Embeddings(),
        # If target text is given, the following will also be populated.
        "target_tokens": lit_types.Tokens(parent="target_text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="target_tokens"),
        "rougeL": lit_types.Scalar(),
    }
    if self.config.output_attention:
      # Add attention for each layer.
      for i in range(self.num_layers):
        spec[f"encoder_layer_{i:d}_attention"] = lit_types.AttentionHeads(
            align=("input_tokens", "input_tokens"))
        spec[f"decoder_layer_{i:d}_attention"] = lit_types.AttentionHeads(
            align=("target_tokens", "target_tokens"))
    return spec
