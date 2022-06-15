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
from transformers import BertTokenizer
from transformers import FlaxBertForQuestionAnswering

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

class T5HFModel(lit_model.Model):
  """T5 using HuggingFace Transformers and Keras.

  This version supports embeddings, attention, and force-decoding of the target
  text, as well as more options to control decoding (such as beam search).
  """

  @property
  def num_layers(self):
    return self.model.config.num_layers

  def __init__(self,
               model_name="mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp",
               model=FlaxBertForQuestionAnswering.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"),
               tokenizer=BertTokenizer.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"),
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
        return_tensors="jax",
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
        # decoder_input_ids=encoded_targets["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        decoder_attention_mask=encoded_targets["attention_mask"])

    model_probs = tf.nn.softmax(results.start_logits, axis=-1)
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
    print('batched_outputs incoming.........')
    print(batched_outputs)
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

    print('preds incoming.........')
    print(preds)
    return preds

class SummarizationWrapper(lit_model.Model):
  """Wrapper class to perform a summarization task."""

  # Mapping from generic T5 fields to this task
  FIELD_RENAMES = {
      "input_text": "context",
      "target_text": "question",
  }
  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, 
              model_name="mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp", 
              model=FlaxBertForQuestionAnswering.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"),
              tokenizer=BertTokenizer.from_pretrained("mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"),
              **config_kw):
    super().__init__()
    self.config = T5ModelConfig(**config_kw)
    self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=False)
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    self.model = model or model_utils.load_pretrained(
        transformers.TFBertForMaskedLM,
        model_name,
        output_hidden_states=True,
        output_attentions=True)

    # # TODO(gehrmann): temp solution for ROUGE.
    # self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # # If output is List[(str, score)] instead of just str
    # self._multi_output = isinstance(self.output_spec()["output_text"],
    #                                 lit_types.GeneratedTextCandidates)
    # self._get_pred_string = (
    #     lit_types.GeneratedTextCandidates.top_text if self._multi_output else
    #     (lambda x: x))

  def preprocess(self, ex: JsonDict) -> JsonDict:
    myquestion = ex['question']
    mycontext = ex["context"]
    ret = myquestion, mycontext
    tokenized_text = self.tokenizer(myquestion, mycontext, return_tensors='jax', padding="longest",
        truncation="longest_first")
    print('tokenized_text incoming.....')
    print(tokenized_text)
    return tokenized_text

  ##
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # If input has a 'tokens' field, use that. Otherwise tokenize the text.
    new_input = []
    for  i in inputs:
        new_input.append(i['question']+ i['context'])

    for i in new_input:
        tokenized_texts = self.tokenizer(i,return_tensors='jax')
    
    encoded_input = self.model(**tokenized_texts)

    print('encoded_input printing below...')
    print(encoded_input)
    # out.logits is a single tensor
    #    <float32>[batch_size, num_tokens, vocab_size]
    # out.hidden_states is a list of num_layers + 1 tensors, each
    #    <float32>[batch_size, num_tokens, h_dim]
    out: transformers.modeling_tf_outputs.TFMaskedLMOutput = \
        self.model(encoded_input)
    batched_outputs = {
        "probas": tf.nn.softmax(out.logits, axis=-1).numpy(),
        "input_ids": encoded_input["input_ids"].numpy(),
        "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1).numpy(),
        # last layer, first token
        "cls_emb": out.hidden_states[-1][:, 0].numpy(),
    }
    # List of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(batched_outputs)
    # Postprocess to remove padding and decode predictions.
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return {
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    spec = spec = {
        "output_text": lit_types.GeneratedText(parent="question")
    }
    spec["rougeL"] = lit_types.Scalar()
    return spec
