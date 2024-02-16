"""LIT model wrappers for generic instrumented Keras LMs."""

import functools
import inspect
import types
from typing import Sequence

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils as lit_utils
import numpy as np
import tensorflow as tf


_DEFAULT_MAX_LENGTH = 1024


class FieldNames(types.SimpleNamespace):
  PROMPT = "prompt"
  RESPONSE = "response"
  PROMPT_EMBEDDINGS = "prompt_embeddings"
  RESPONSE_EMBEDDINGS = "response_embeddings"
  TARGET = "target"
  TOKENS = "tokens"
  TARGET_MASK = "target_mask"
  GRAD_DOT_INPUT = "grad_dot_input"
  GRAD_NORM = "grad_l2"
  TOKEN_LOSS = "token_loss"


class _KerasBaseModel(lit_model.BatchedModel):
  """Base LIT model wrapper class for Keras on TensorFlow."""

  # TODO(lit-dev): pytype annotations for model= ?
  # Should be keras_nlp.models.generative_task.GenerativeTask
  def __init__(
      self,
      model,
      max_length: int = _DEFAULT_MAX_LENGTH,
      dynamic_sequence_length: bool = True,
      batch_size: int = 16,
  ):
    """Base wrapper for a Keras/TF2 LM supporting the layer_intercept_fn API.

    Model should support the following methods:
    - .generate()
    - .score()*
    - .preprocessor.generate_preprocess()
    . .preprocessor.tokenizer.id_to_token()
    . .backbone.token_embedding()

    * The score function should accept layer_intercept_fn= as a way to intercept
    and manipulate activations between layers. We use this for salience, below.

    Args:
      model: pre-loaded Keras LM using the TF backend
      max_length: max sequence length
      dynamic_sequence_length: if true, will trim padding to the length of the
        longest sequence in a batch. Recommended for CPU and GPU usage, but may
        be disabled for compilation where a fixed shape is required.
      batch_size: batch size
    """
    super().__init__()

    self.model = model
    self.batch_size = batch_size
    self.max_length = max_length
    self.dynamic_sequence_length = dynamic_sequence_length

    self.ids_to_tokens = np.vectorize(
        self.model.preprocessor.tokenizer.id_to_token
    )

    # map ids: <tf.int>[batch_size, num_tokens]
    # to embs: <tf.float>[batch_size, num_tokens, emb_dim]
    self.embedder = self.model.backbone.token_embedding

  def encode_inputs(self, texts: Sequence[str]):
    """Encode inputs, with optional dynamic trimming.

    By default, the model's generate_preprocess() pads to a fixed sequence
    length, either specified as sequence_length= or using an internal default.

    Here, we optionally trim this to remove extraneous padding positions based
    on the actual contents of the minibatch. This can greatly speed up
    performance when running on CPU or GPU.

    Args:
      texts: list of input strings

    Returns:
      encoded_inputs compatible with model.score() or other functions
    """
    # First: pack to max_length
    encoded_inputs = self.model.preprocessor.generate_preprocess(
        texts, sequence_length=self.max_length
    )
    if not self.dynamic_sequence_length:
      return encoded_inputs

    # Trim to the maximum length needed to contain any non-padding tokens.
    mask = encoded_inputs["padding_mask"]
    # Find position of last 'True' in each row.
    seq_ends: Sequence[int] = [
        1 + tf.reduce_max(tf.where(mask[i])).numpy().tolist()
        for i in range(mask.shape[0])
    ]
    trimmed_length = max(seq_ends)
    # TODO(lit-dev): remove this line, or make it logging.debug ?
    logging.info(
        "Trimming batch to trimmed_length = %d based on sequence ends %s",
        trimmed_length,
        seq_ends,
    )
    # Actually trim the input tensors.
    return {k: v[:, :trimmed_length] for k, v in encoded_inputs.items()}

  @classmethod
  def from_loaded(cls, existing: "_KerasBaseModel", *args, **kw):
    """Share weights and underlying Keras model with another instance."""
    return cls(model=existing.model, *args, **kw)

  def max_minibatch_size(self) -> int:
    return self.batch_size

  @classmethod
  def init_spec(cls):
    # Cannot initialize from spec, because we need a Keras model object.
    return None

  def input_spec(self):
    return {
        FieldNames.PROMPT: lit_types.TextSegment(),
        FieldNames.TARGET: lit_types.TextSegment(required=False),
    }


class KerasGenerationModel(_KerasBaseModel):
  """LIT model wrapper for generating text with Keras on TensorFlow.

  This class accepts a loaded model and provides the LIT-required functions plus
  additional helper functions for generation tasks.

  This class supports generation and pass-through modes. If a dataset provides a
  pre-populated 'response' column then this model will return that text instead
  of generating new text from the 'prompt'. This allows the same model wrapper
  to be efficiently used to examine saved results from bulk-inference pipelines
  and new generations from, e.g., counterfactually generated examples, or novel
  evaluation datasets.
  """

  def __init__(self, *args, output_embeddings=True, **kw):
    super().__init__(*args, **kw)
    self.output_embeddings = output_embeddings

  def embed_texts(self, texts: Sequence[str]):
    processed_inputs = self.encode_inputs(texts)
    # <tf.float>[batch_size, num_tokens, emb_dim]
    embs = self.embedder(processed_inputs["token_ids"])
    # <tf.bool>[batch_size, num_tokens]
    mask = processed_inputs["padding_mask"]
    return embs, mask

  def embed_and_mean_pool(self, texts: Sequence[str]):
    """Return a single vector for each text."""
    embs, mask = self.embed_texts(texts)
    # <tf.float>[batch_size, num_tokens, 1]
    mask = tf.expand_dims(tf.cast(mask, dtype=embs.dtype), axis=2)
    # <tf.float>[batch_size, 1, emb_dim]
    pooled_embs = tf.reduce_sum(
        mask * embs, axis=1, keepdims=True
    ) / tf.reduce_sum(mask, axis=1, keepdims=True)
    # <tf.float>[batch_size, emb_dim]
    return tf.squeeze(pooled_embs, axis=1)

  def predict_minibatch(
      self,
      inputs: list[lit_types.JsonDict],
  ) -> list[lit_types.JsonDict]:
    prompts: Sequence[str] = [ex[FieldNames.PROMPT] for ex in inputs]

    # TODO(lit-dev): suppport loading cached responses here, since running
    # generation can be expensive.
    full_responses: Sequence[str] = list(
        self.model.generate(prompts, max_length=self.max_length)
    )
    # Model outputs include the prompt, so trim that off and just return the
    # generated portion.
    responses: Sequence[str] = [
        response[len(prompt) :]
        for response, prompt in zip(full_responses, prompts)
    ]

    outputs = [{FieldNames.RESPONSE: response} for response in responses]

    if self.output_embeddings:
      prompt_embeddings = self.embed_and_mean_pool(prompts)
      # TODO(lit-dev): embed prompt + response and trim embedding instead?
      # Or just embed full_response.
      response_embeddings = self.embed_and_mean_pool(responses)

      for i in range(len(inputs)):
        outputs[i][FieldNames.PROMPT_EMBEDDINGS] = prompt_embeddings[i].numpy()
        outputs[i][FieldNames.RESPONSE_EMBEDDINGS] = response_embeddings[
            i
        ].numpy()

    return outputs

  def output_spec(self) -> lit_types.Spec:
    ret = {
        FieldNames.RESPONSE: lit_types.GeneratedText(parent=FieldNames.TARGET)
    }
    if self.output_embeddings:
      return ret | {
          FieldNames.PROMPT_EMBEDDINGS: lit_types.Embeddings(),
          FieldNames.RESPONSE_EMBEDDINGS: lit_types.Embeddings(),
      }
    return ret


class KerasSalienceModel(_KerasBaseModel):
  """LIT model wrapper for computing salience with Keras on TensorFlow.

  This class accepts a loaded model and provides the LIT-required functions plus
  additional helper functions to convert and clean tokens and to compute
  sequence salience.

  This class does not support generation; use the KerasGenerationModel class to
  generate the text for which this class will compute salience.
  """

  def __init__(self, *args, **kw):
    super().__init__(*args, **kw)

    score_fn = getattr(self.model, "score", None)

    if score_fn is None or not inspect.ismethod(score_fn):
      raise TypeError(
          "Salience is computed via a .score() API, which is not supported by "
          "all GenerativeTask models in KerasNLP. Please provide a model that "
          "supports this API."
      )

  def _pred(self, input_ids, padding_mask, target_masks):
    """Predict a batch of tokenized text."""
    # <tf.int>[batch_size, num_tokens]; ignore the last one in each row.
    target_ids = tf.roll(input_ids, shift=-1, axis=1)

    ##
    # Process target masks

    # It doesn't make sense to interpret the first token, since it is not ever
    # predicted. But we need to ensure that the mask[0] is zero, so it doesn't
    # cause problems when 'rolled' to the last position below.
    modified_masks = [[0] + list(mask[1:]) for mask in target_masks]
    seq_len = target_ids.shape[1]
    pad_fn = functools.partial(
        lit_utils.pad1d,
        min_len=seq_len,
        max_len=seq_len,
        pad_val=0,
        pad_left=False,
    )
    padded_target_masks = np.stack(
        [pad_fn(mask) for mask in modified_masks],
        axis=0,
    )

    padded_target_masks = tf.constant(padded_target_masks, dtype=tf.bool)
    # Shift masks back so they align with target_ids.
    loss_mask = tf.roll(padded_target_masks, shift=-1, axis=1)

    embeddings = None

    with tf.GradientTape(watch_accessed_variables=False) as tape:

      def layer_intercept_fn(x, i):
        if i == -1:
          nonlocal embeddings, tape
          embeddings = x
          tape.watch(embeddings)
        return x

      # <tf.float>[batch_size, num_tokens]
      per_token_loss = self.model.score(
          token_ids=input_ids,
          padding_mask=padding_mask,
          scoring_mode="loss",
          layer_intercept_fn=layer_intercept_fn,
          target_ids=target_ids,
      )
      masked_loss = per_token_loss * tf.cast(loss_mask, per_token_loss.dtype)

    # <tf.float>[batch_size, num_tokens, hdim]
    grads = tape.gradient(masked_loss, embeddings)
    # <tf.float>[batch_size, num_tokens]
    grad_l2 = tf.norm(grads, axis=2)
    # <tf.float>[batch_size, num_tokens]
    grad_dot_input = tf.reduce_sum(grads * embeddings, axis=2)

    batched_outputs = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        # Gradients are already aligned to input tokens.
        FieldNames.GRAD_NORM: grad_l2,
        FieldNames.GRAD_DOT_INPUT: grad_dot_input,
        # Shift token loss to align with (input) tokens.
        # FieldNames.TOKEN_LOSS: tf.roll(per_token_loss, shift=1, axis=1),
    }

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    mask = preds.pop("padding_mask").astype(bool)
    ids = preds.pop("input_ids")[mask]
    preds[FieldNames.TOKENS] = self.ids_to_tokens(ids)
    for key in lit_utils.find_spec_keys(
        self.output_spec(), lit_types.TokenScores
    ):
      preds[key] = preds[key][mask]
    # First token (<bos>) is not actually predicted, so return 0 for loss.
    # preds[FieldNames.TOKEN_LOSS][0] = 0

    return preds

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    texts: Sequence[str] = [
        ex[FieldNames.PROMPT] + ex.get(FieldNames.TARGET, "") for ex in inputs
    ]
    preprocessed_texts = self.encode_inputs(texts)
    sequence_ids = preprocessed_texts["token_ids"]
    padding_mask = preprocessed_texts["padding_mask"]

    target_masks = [ex.get(FieldNames.TARGET_MASK, []) for ex in inputs]

    # Get the predictions.
    batched_outputs = self._pred(sequence_ids, padding_mask, target_masks)
    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = lit_utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return super().input_spec() | {
        FieldNames.TARGET_MASK: lit_types.TokenScores(align="", required=False),
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        FieldNames.TOKENS: lit_types.Tokens(parent=""),  # All tokens.
        FieldNames.GRAD_NORM: lit_types.TokenScores(align=FieldNames.TOKENS),
        FieldNames.GRAD_DOT_INPUT: lit_types.TokenScores(
            align=FieldNames.TOKENS
        ),
        # FieldNames.TOKEN_LOSS: lit_types.TokenScores(align=FieldNames.TOKENS),
    }


class KerasTokenizerModel(_KerasBaseModel):
  """LIT model wrapper for tokenizing text with Keras on TensorFlow.

  This class accepts a loaded model and provides the LIT-required functions plus
  additional helper functions to convert and clean tokens.
  """

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Be sure to cast to bool, otherwise this will select intger positions 0, 1
    # rather than acting as a boolean mask.
    mask = preds.pop("padding_mask").astype(bool)
    ids = preds.pop("token_ids")[mask]
    preds[FieldNames.TOKENS] = self.ids_to_tokens(ids)
    return preds

  def predict_minibatch(self, inputs):
    """Tokenize a single minibatch of examples."""
    texts: Sequence[str] = [
        ex[FieldNames.PROMPT] + ex.get(FieldNames.TARGET, "") for ex in inputs
    ]
    preprocessed_texts = self.encode_inputs(texts)
    batched_outputs = {
        "token_ids": preprocessed_texts["token_ids"],
        "padding_mask": preprocessed_texts["padding_mask"],
    }
    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = lit_utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def output_spec(self) -> lit_types.Spec:
    return {
        FieldNames.TOKENS: lit_types.Tokens(parent=""),  # All tokens.
    }


def initialize_model_group_for_salience(
    name, *args, **kw
) -> dict[str, lit_model.Model]:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  generation_model = KerasGenerationModel(*args, **kw)
  salience_model = KerasSalienceModel(*args, **kw)
  tokenizer_model = KerasTokenizerModel(*args, **kw)
  return {
      name: generation_model,
      f"_{name}_salience": salience_model,
      f"_{name}_tokenizer": tokenizer_model,
  }
