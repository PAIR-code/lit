# Lint as: python3
"""LIT wrappers for T5, supporting both HuggingFace and SavedModel formats."""
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils

import tensorflow as tf
# tensorflow_text is required for T5 SavedModel
import tensorflow_text  # pylint: disable=unused-import
import transformers

from rouge_score import rouge_scorer

JsonDict = lit_types.JsonDict


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
  # Input options
  inference_batch_size: int = 4
  # Generation options
  beam_size: int = 4
  max_gen_length: int = 50
  num_to_generate: int = 1
  # Decoding options
  token_top_k: int = 10
  output_attention: bool = False


def validate_t5_model(model: lit_model.Model) -> lit_model.Model:
  """Validate that a given model looks like a T5 model.

  This checks the model spec at runtime; it is intended to be used before server
  start, such as in the __init__() method of a wrapper class.

  Args:
    model: a LIT model

  Returns:
    model: the same model

  Raises:
    AssertionError: if the model's spec does not match that expected for a T5
    model.
  """
  # Check inputs
  ispec = model.input_spec()
  assert "input_text" in ispec
  assert isinstance(ispec["input_text"], lit_types.TextSegment)
  if "target_text" in ispec:
    assert isinstance(ispec["target_text"], lit_types.TextSegment)

  # Check outputs
  ospec = model.output_spec()
  assert "output_text" in ospec
  assert isinstance(
      ospec["output_text"],
      (lit_types.GeneratedText, lit_types.GeneratedTextCandidates))
  assert ospec["output_text"].parent == "target_text"

  return model


class T5SavedModel(lit_model.Model):
  """T5 from a TensorFlow SavedModel, for black-box access.

  To create a SavedModel from a regular T5 checkpoint, see
  https://github.com/google-research/text-to-text-transfer-transformer#export
  """

  def __init__(self, saved_model_path: str, model=None, **config_kw):
    super().__init__()
    # By default, SavedModels from the original T5 codebase have batch_size=1
    # hardcoded. Use setdefault here so that the user can still override if
    # they've fixed this upstream.
    config_kw.setdefault("inference_batch_size", 1)
    self.config = T5ModelConfig(**config_kw)
    self.model = model or tf.saved_model.load(saved_model_path)

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return self.config.inference_batch_size

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    model_inputs = tf.constant([ex["input_text"] for ex in inputs])
    model_outputs = self.model.signatures["serving_default"](model_inputs)
    return [{
        "output_text": m.decode("utf-8")
    } for m in model_outputs["outputs"].numpy()]

  def input_spec(self):
    return {
        "input_text": lit_types.TextSegment(),
        "target_text": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    return {"output_text": lit_types.GeneratedText(parent="target_text")}


class T5HFModel(lit_model.Model):
  """T5 using HuggingFace Transformers and Keras.

  This version supports embeddings, attention, and force-decoding of the target
  text, as well as more options to control decoding (such as beam search).
  """

  @property
  def num_layers(self):
    return self.model.config.num_layers

  def __init__(self,
               model_name="t5-small",
               model=None,
               tokenizer=None,
               **config_kw):
    super().__init__()
    self.config = T5ModelConfig(**config_kw)
    assert self.config.num_to_generate <= self.config.beam_size
    self.tokenizer = tokenizer or transformers.T5Tokenizer.from_pretrained(
        model_name)
    self.model = model or model_utils.load_pretrained(
        transformers.TFT5ForConditionalGeneration,
        model_name,
        output_hidden_states=True,
        output_attentions=self.config.output_attention)

  def _encode_texts(self, texts: List[str]):
    return self.tokenizer.batch_encode_plus(
        texts,
        return_tensors="tf",
        padding="longest",
        truncation="longest_first")

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
        input_ids=encoded_inputs["input_ids"],
        decoder_input_ids=encoded_targets["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        decoder_attention_mask=encoded_targets["attention_mask"])

    model_probs = tf.nn.softmax(results.logits, axis=-1)
    top_k = tf.math.top_k(
        model_probs, k=self.config.token_top_k, sorted=True, name=None)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "input_ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
        "target_ids": encoded_targets["input_ids"],
        "target_ntok": tf.reduce_sum(encoded_targets["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }
    # encoder_last_hidden_state is <float>[batch_size, num_tokens, emb_dim]
    # take the mean over real tokens to get <float>[batch_size, emb_dim]
    batched_outputs["encoder_final_embedding"] = masked_token_mean(
        results.encoder_last_hidden_state, encoded_inputs["attention_mask"])

    if self.config.output_attention:
      for i in range(len(results.decoder_attentions)):
        batched_outputs[
            f"decoder_layer_{i+1:d}_attention"] = results.decoder_attentions[i]
      for i in range(len(results.encoder_attentions)):
        batched_outputs[
            f"encoder_layer_{i+1:d}_attention"] = results.encoder_attentions[i]

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
    candidates = [
        self.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in preds.pop("generated_ids")
    ]
    if self.config.num_to_generate > 1:
      preds["output_text"] = [(s, None) for s in candidates]
    else:
      preds["output_text"] = candidates[0]

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

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return self.config.inference_batch_size

  def predict_minibatch(self, inputs):
    """Run model on a single batch.

    Args:
      inputs: List[Dict] with fields as described by input_spec()

    Returns:
      outputs: List[Dict] with fields as described by output_spec()
    """
    # Text as sequence of sentencepiece ID"s.
    encoded_inputs = self._encode_texts([ex["input_text"] for ex in inputs])
    encoded_targets = self._encode_texts(
        [ex.get("target_text", "") for ex in inputs])

    ##
    # Force-decode on target text, and also get encoder embs and attention.
    batched_outputs = self._force_decode(encoded_inputs, encoded_targets)
    # Get the conditional generation from the model.
    # Workaround for output_hidden not being compatible with generate.
    # See https://github.com/huggingface/transformers/issues/8361
    self.model.config.output_hidden_states = False
    generated_ids = self.model.generate(
        encoded_inputs.input_ids,
        num_beams=self.config.beam_size,
        attention_mask=encoded_inputs.attention_mask,
        max_length=self.config.max_gen_length,
        num_return_sequences=self.config.num_to_generate)
    # [batch_size*num_return_sequences, num_steps]
    # -> [batch_size, num_return_sequences, num_steps]
    batched_outputs["generated_ids"] = tf.reshape(
        generated_ids,
        [-1, self.config.num_to_generate, generated_ids.shape[-1]])
    self.model.config.output_hidden_states = True

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return list(map(self._postprocess, unbatched_outputs))

  def input_spec(self):
    return {
        "input_text": lit_types.TextSegment(),
        "target_text": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    spec = {
        "output_text": lit_types.GeneratedText(parent="target_text"),
        "input_tokens": lit_types.Tokens(parent="input_text"),
        "encoder_final_embedding": lit_types.Embeddings(),
        # If target text is given, the following will also be populated.
        "target_tokens": lit_types.Tokens(parent="target_text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="target_tokens"),
    }
    if self.config.num_to_generate > 1:
      spec["output_text"] = lit_types.GeneratedTextCandidates(
          parent="target_text")

    if self.config.output_attention:
      # Add attention for each layer.
      for i in range(self.num_layers):
        spec[f"encoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="input_tokens", align_out="input_tokens")
        spec[f"decoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="target_tokens", align_out="target_tokens")
    return spec


##
# Task-specific wrapper classes.


class TranslationWrapper(lit_model.ModelWrapper):
  """Wrapper class for machine translation."""

  # Mapping from generic T5 fields to this task
  FIELD_RENAMES = {
      "input_text": "source",
      "target_text": "target",
      "output_text": "translation",
  }

  # From Appendix D of https://arxiv.org/pdf/1910.10683.pdf.
  # Add more of these if your model supports them.
  LANGCODE_TO_NAME = {
      "en": "English",
      "de": "German",
      "fr": "French",
      "ro": "Romanian",
  }

  INPUT_TEMPLATE = "translate {source_language} to {target_language}: {source}"

  def __init__(self, model: lit_model.Model):
    model = validate_t5_model(model)
    super().__init__(model)

  def preprocess(self, ex: JsonDict) -> JsonDict:
    input_kw = {
        "source_language": self.LANGCODE_TO_NAME[ex["source_language"]],
        "target_language": self.LANGCODE_TO_NAME[ex["target_language"]],
        "source": ex["source"]
    }
    ret = {"input_text": self.INPUT_TEMPLATE.format(**input_kw)}
    if "target" in ex:
      ret["target_text"] = ex["target"]
    return ret

  ##
  # LIT API implementation
  def description(self) -> str:
    return "T5 for machine translation\n" + self.wrapped.description()

  # TODO(b/170662608): remove these after batching API is cleaned up.
  def max_minibatch_size(self) -> int:
    raise NotImplementedError("Use predict() instead.")

  def predict_minibatch(self, inputs):
    raise NotImplementedError("Use predict() instead.")

  def predict(self, inputs):
    """Predict on a single minibatch of examples."""
    model_inputs = (self.preprocess(ex) for ex in inputs)
    outputs = self.wrapped.predict(model_inputs)
    return (utils.remap_dict(mo, self.FIELD_RENAMES) for mo in outputs)

  def predict_with_metadata(self, indexed_inputs):
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex["data"] for ex in indexed_inputs))

  def input_spec(self):
    spec = lit_types.remap_spec(self.wrapped.input_spec(), self.FIELD_RENAMES)
    spec["source_language"] = lit_types.CategoryLabel()
    spec["target_language"] = lit_types.CategoryLabel()
    return spec

  def output_spec(self):
    return lit_types.remap_spec(self.wrapped.output_spec(), self.FIELD_RENAMES)


class SummarizationWrapper(lit_model.ModelWrapper):
  """Wrapper class to perform a summarization task."""

  # Mapping from generic T5 fields to this task
  FIELD_RENAMES = {
      "input_text": "document",
      "target_text": "reference",
  }

  def __init__(self, model: lit_model.Model):
    model = validate_t5_model(model)
    super().__init__(model)

    # TODO(gehrmann): temp solution for ROUGE.
    self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # If output is List[(str, score)] instead of just str
    self._multi_output = isinstance(self.output_spec()["output_text"],
                                    lit_types.GeneratedTextCandidates)
    self._get_pred_string = (
        lit_types.GeneratedTextCandidates.top_text if self._multi_output else
        (lambda x: x))

  def preprocess(self, ex: JsonDict) -> JsonDict:
    ret = {"input_text": "summarize: " + ex["document"]}
    if "reference" in ex:
      ret["target_text"] = ex["reference"]
    return ret

  ##
  # LIT API implementation
  def description(self) -> str:
    return "T5 for summarization\n" + self.wrapped.description()

  # TODO(b/170662608): remove these after batching API is cleaned up.
  def max_minibatch_size(self) -> int:
    raise NotImplementedError("Use predict() instead.")

  def predict_minibatch(self, inputs):
    raise NotImplementedError("Use predict() instead.")

  def predict(self, inputs):
    """Predict on a single minibatch of examples."""
    inputs = list(inputs)  # needs to be referenced below, so keep full list
    model_inputs = (self.preprocess(ex) for ex in inputs)
    outputs = self.wrapped.predict(model_inputs)
    outputs = (utils.remap_dict(mo, self.FIELD_RENAMES) for mo in outputs)

    # TODO(gehrmann): temp solution to get ROUGE scores in data table.
    for ex, mo in zip(inputs, outputs):
      score = self._scorer.score(
          target=ex["reference"],
          prediction=self._get_pred_string(mo["output_text"]))
      mo["rougeL"] = float(score["rougeL"].fmeasure)
      yield mo

  def predict_with_metadata(self, indexed_inputs):
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex["data"] for ex in indexed_inputs))

  def input_spec(self):
    return lit_types.remap_spec(self.wrapped.input_spec(), self.FIELD_RENAMES)

  def output_spec(self):
    spec = lit_types.remap_spec(self.wrapped.output_spec(), self.FIELD_RENAMES)
    spec["rougeL"] = lit_types.Scalar()
    return spec
