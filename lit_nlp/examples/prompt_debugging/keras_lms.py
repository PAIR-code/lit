"""LIT model wrappers for generic instrumented Keras LMs."""

from collections.abc import Sequence
import functools
import inspect
from typing import Optional

from absl import logging
import keras
from keras_nlp import models as keras_models
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.prompt_debugging import constants as pd_constants
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils as lit_utils


# pylint: disable=g-import-not-at-top
# pytype: disable=import-error
# NOTE: The Keras backend must be set before loading the Keras library. You can
# set the backend using the KERAS_BACKEND environment variable or your
# ~/.keras/keras.json configuration file. For more information, see:
# https://keras.io/getting_started/#configuring-your-backend
if keras.backend.backend() == "tensorflow":
  import tensorflow as tf
elif keras.backend.backend() == "torch":
  import torch
else:
  # TODO(b/333373960): Update imports once a JAX salience is supported.
  raise ValueError(f"Unsupported backend: {keras.backend.backend()}")
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top


_DEFAULT_MAX_LENGTH = 1024


class _KerasBaseModel(lit_model.BatchedModel):
  """Base LIT model wrapper class for Keras on TensorFlow."""

  # TODO(lit-dev): pytype annotations for model= ?
  # Should be keras_nlp.models.generative_task.GenerativeTask
  def __init__(
      self,
      model: Optional[keras_models.CausalLM] = None,
      model_name_or_path: Optional[str] = None,
      max_length: int = _DEFAULT_MAX_LENGTH,
      dynamic_sequence_length: bool = True,
      batch_size: int = 16,
  ):
    """Base wrapper for a Keras CausalLM supporting the layer_intercept_fn API.

    Model should support the following methods:
    - .generate()
    - .score()*
    - .preprocessor.generate_preprocess()
    . .preprocessor.tokenizer.id_to_token()
    . .backbone.token_embedding()

    * The score function should accept layer_intercept_fn= as a way to intercept
    and manipulate activations between layers. We use this for salience, below.

    Args:
      model: A pre-loaded Keras CausalLM, prioritized over model_name_or_path.
      model_name_or_path: A URL, path, or preset name for the model to load,
      max_length: max sequence length
      dynamic_sequence_length: if true, will trim padding to the length of the
        longest sequence in a batch. Recommended for CPU and GPU usage, but may
        be disabled for compilation where a fixed shape is required.
      batch_size: batch size
    """
    super().__init__()

    if model is not None:
      self.model = model
    elif model_name_or_path is not None:
      if (
          is_tar_gz := model_name_or_path.endswith(".tar.gz")
      ) or file_cache.is_remote(model_name_or_path):
        model_name_or_path = file_cache.cached_path(
            model_name_or_path,
            extract_compressed_file=is_tar_gz,
        )
      self.model = keras_models.CausalLM.from_preset(model_name_or_path)
    else:
      raise ValueError("Must provide either model or model_name_or_path.")

    self.batch_size = batch_size
    self.max_length = max_length
    self.dynamic_sequence_length = dynamic_sequence_length

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
      A dict[str, Tensor] compatible with model.score(), etc. functions.
    """
    # First: pack to max_length
    encoded_inputs = self.model.preprocessor.generate_preprocess(
        texts, sequence_length=self.max_length
    )
    if not self.dynamic_sequence_length:
      return encoded_inputs

    # Trim to the maximum length needed to contain any non-padding tokens.
    mask = encoded_inputs["padding_mask"]

    if keras.backend.backend() == "tensorflow":
      max_indices = [tf.reduce_max(tf.where(row)) for row in mask]
    elif keras.backend.backend() == "torch":
      max_indices = [torch.max(torch.where(row)[0]) for row in mask]
    else:
      raise ValueError(f"Unsupported backend: {keras.backend.backend()}")
    # Find position of last 'True' in each row.
    seq_ends: Sequence[int] = [
        keras.ops.convert_to_numpy(i).tolist() + 1 for i in max_indices
    ]
    longest_sequence = max(seq_ends)
    # TODO(lit-dev): remove this line, or make it logging.debug ?
    logging.info(
        "Trimming batch to trimmed_length = %d based on sequence ends %s",
        longest_sequence,
        seq_ends,
    )
    # Actually trim the input tensors.
    return {k: v[:, :longest_sequence] for k, v in encoded_inputs.items()}

  def clean_subword_token(self, tok: str) -> str:
    """Clean up special subword token from the tokenizers if necessary.

    Args:
      tok: the token to clean up.
    Returns:
      The replaced token if the provided token matches the special subword token
      below; otherwise, the original token is returned.
    """
    # For GPT2 tokenizer.
    tok = tok.replace("Ċ", "\n")  # newlines
    tok = tok.replace("Ġ", "▁")  # start of word -> magic underscore
    # For SentencePiece Tokenizer.
    tok = tok.replace("<0x0A>", "\n")  # newlines
    return tok

  def ids_to_clean_tokens(self, ids: Sequence[int]) -> Sequence[str]:
    return [
        self.clean_subword_token(
            self.model.preprocessor.tokenizer.id_to_token(id)
        )
        for id in ids
    ]

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
    return pd_constants.INPUT_SPEC


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
    # <float>[batch_size, num_tokens, emb_dim]
    embs = self.embedder(processed_inputs["token_ids"])
    # <bool>[batch_size, num_tokens]
    mask = processed_inputs["padding_mask"]
    return embs, mask

  def embed_and_mean_pool(self, texts: Sequence[str]):
    """Return a single vector for each text."""
    embs, mask = self.embed_texts(texts)
    # <float>[batch_size, num_tokens, 1]
    cast_mask = keras.ops.cast(mask, dtype=embs.dtype)

    if keras.backend.backend() == "tensorflow":
      expanded_mask = tf.expand_dims(cast_mask, axis=2)
      pooled_embs = tf.reduce_sum(
          expanded_mask * embs, axis=1, keepdims=True
      ) / tf.reduce_sum(expanded_mask, axis=1, keepdims=True)
      return tf.squeeze(pooled_embs, axis=1)
    elif keras.backend.backend() == "torch":
      expanded_mask = torch.unsqueeze(cast_mask, dim=2)
      pooled_embs = torch.sum(
          expanded_mask * embs, dim=1, keepdim=True
      ) / torch.sum(expanded_mask, dim=1, keepdim=True)
      return torch.squeeze(pooled_embs, dim=1)
    else:
      raise ValueError(f"Unsupported backend: {keras.backend.backend()}")

  def predict_minibatch(
      self,
      inputs: list[lit_types.JsonDict],
  ) -> list[lit_types.JsonDict]:
    prompts: Sequence[str] = [
        ex[pd_constants.FieldNames.PROMPT] for ex in inputs
    ]

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

    outputs = [
        {pd_constants.FieldNames.RESPONSE: response} for response in responses
    ]

    if self.output_embeddings:
      prompt_embeddings = self.embed_and_mean_pool(prompts)
      # TODO(lit-dev): embed prompt + response and trim embedding instead?
      # Or just embed full_response.
      response_embeddings = self.embed_and_mean_pool(responses)

      for o, p, r in zip(outputs, prompt_embeddings, response_embeddings):
        o[pd_constants.FieldNames.PROMPT_EMBEDDINGS] = (
            keras.ops.convert_to_numpy(p)
        )
        o[pd_constants.FieldNames.RESPONSE_EMBEDDINGS] = (
            keras.ops.convert_to_numpy(r)
        )

    return outputs

  def output_spec(self) -> lit_types.Spec:
    ret = pd_constants.OUTPUT_SPEC_GENERATION
    if self.output_embeddings:
      return ret | pd_constants.OUTPUT_SPEC_GENERATION_EMBEDDINGS
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
          "all KerasNLP CausalLM models. Please provide a model that "
          "supports this API."
      )

  def _pred(self, input_ids, padding_mask, target_masks):
    """Predict a batch of tokenized text.

    Args:
      input_ids: A Tensor with shape <int>[batch_size, num_tokens]
      padding_mask: A Tensor with shape <int>[batch_size, num_tokens]
      target_masks: A Numpy Array with shape <bool>[batch_size, num_tokens]

    Returns:
      Batched outputs for post-processing.
    """
    ##
    # Process target masks
    # It doesn't make sense to interpret the first token, since it is not ever
    # predicted. But we need to ensure that the mask[0] is zero, so it doesn't
    # cause problems when 'rolled' to the last position below.
    seq_len = keras.ops.shape(input_ids)[1]
    pad_fn = functools.partial(
        lit_utils.pad1d,
        min_len=seq_len,
        max_len=seq_len,
        pad_val=0,
        pad_left=False,
    )

    modified_masks = [[0] + list(mask[1:]) for mask in target_masks]
    stacked_padded_masks = keras.ops.stack(
        [pad_fn(mask) for mask in modified_masks],
        axis=0,
    )
    # Shift masks back so they align with the target_ids generated in the
    # backend-specific prediction functions.
    rolled_masks = keras.ops.roll(stacked_padded_masks, shift=-1, axis=1)
    loss_mask = keras.ops.convert_to_tensor(rolled_masks, dtype="bool")

    pred_kw_args = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "loss_mask": loss_mask,
    }
    if keras.backend.backend() == "tensorflow":
      grad_l2, grad_dot_input = self._pred_tf(**pred_kw_args)
    elif keras.backend.backend() == "jax":
      grad_l2, grad_dot_input = self._pred_jax(**pred_kw_args)
    elif keras.backend.backend() == "torch":
      grad_l2, grad_dot_input = self._pred_torch(**pred_kw_args)
    else:
      raise ValueError(f"Unsupported backend: {keras.backend.backend()}")

    batched_outputs = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        pd_constants.FieldNames.GRAD_NORM: grad_l2,
        pd_constants.FieldNames.GRAD_DOT_INPUT: grad_dot_input,
    }

    return batched_outputs

  def _pred_tf(self, input_ids, padding_mask, loss_mask):
    # <int>[batch_size, num_tokens]; ignore the last one in each row.
    target_ids = tf.roll(input_ids, shift=-1, axis=1)
    embeddings = None

    with tf.GradientTape(watch_accessed_variables=False) as tape:

      def layer_intercept_fn(x, i):
        if i == -1:
          nonlocal embeddings, tape
          embeddings = x
          tape.watch(embeddings)
        return x

      # <float>[batch_size, num_tokens]
      per_token_loss = self.model.score(
          token_ids=input_ids,
          padding_mask=padding_mask,
          scoring_mode="loss",
          layer_intercept_fn=layer_intercept_fn,
          target_ids=target_ids,
      )
      masked_loss = per_token_loss * keras.ops.cast(
          loss_mask, per_token_loss.dtype
      )

    # <float>[batch_size, num_tokens, hdim]
    grads = tape.gradient(masked_loss, embeddings)
    # <float>[batch_size, num_tokens]
    grad_l2 = tf.norm(grads, axis=2)
    # <float>[batch_size, num_tokens]
    grad_dot_input = tf.reduce_sum(grads * embeddings, axis=2)
    return grad_l2, grad_dot_input

  # TODO(b/333373960): Implement salience computation for JAX.
  def _pred_jax(self, input_ids, padding_mask, loss_mask):
    # NOTE: JAX computes gradients automatically w.r.t function inputs and
    # outputs. The score function takes token_ids as its input but salience is
    # computed w.r.t. the embeddings, thus JAX cannot differentiate the loss
    # w.r.t. the embeddings and taking gradients w.r.t. the token_ids is not
    # equivalent. For now, we raise an error if using JAX.
    raise NotImplementedError("JAX backend not supported for salience.")

  def _pred_torch(self, input_ids, padding_mask, loss_mask):
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    embeddings = None

    def layer_intercept_fn(x, i):
      if i == -1:
        nonlocal embeddings
        embeddings = x
      return x

    per_token_loss = self.model.score(
        token_ids=input_ids,
        padding_mask=padding_mask,
        scoring_mode="loss",
        layer_intercept_fn=layer_intercept_fn,
        target_ids=target_ids,
    )

    if embeddings is None:
      raise ValueError("Embeddings are None after scoring.")

    masked_loss = per_token_loss * keras.ops.cast(
        loss_mask, per_token_loss.dtype
    )

    # <float>[batch_size, num_tokens, hdim]
    grads = torch.autograd.grad(
        masked_loss, embeddings, grad_outputs=torch.ones_like(masked_loss)
    )[0]
    embeddings = embeddings.detach()
    # <float>[batch_size, num_tokens]
    grad_l2 = torch.norm(grads, dim=2)
    # <float>[batch_size, num_tokens]
    grad_dot_input = torch.sum(grads * embeddings, dim=2)
    return grad_l2, grad_dot_input

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    mask = preds.pop("padding_mask").astype(bool)
    ids = preds.pop("input_ids")[mask]
    preds[pd_constants.FieldNames.TOKENS] = self.ids_to_clean_tokens(ids)
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
        ex[pd_constants.FieldNames.PROMPT]
        + ex.get(pd_constants.FieldNames.TARGET, "")
        for ex in inputs
    ]
    preprocessed_texts = self.encode_inputs(texts)
    sequence_ids = preprocessed_texts["token_ids"]
    padding_mask = preprocessed_texts["padding_mask"]

    target_masks = [
        ex.get(pd_constants.FieldNames.TARGET_MASK, []) for ex in inputs
    ]

    # Get the predictions.
    batched_outputs = self._pred(sequence_ids, padding_mask, target_masks)
    # Convert to numpy for post-processing.
    detached_outputs = {
        k: keras.ops.convert_to_numpy(v) for k, v in batched_outputs.items()
    }
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = lit_utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return super().input_spec() | pd_constants.INPUT_SPEC_SALIENCE

  def output_spec(self) -> lit_types.Spec:
    return pd_constants.OUTPUT_SPEC_SALIENCE


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
    preds[pd_constants.FieldNames.TOKENS] = self.ids_to_clean_tokens(ids)
    return preds

  def predict_minibatch(self, inputs):
    """Tokenize a single minibatch of examples."""
    texts: Sequence[str] = [
        ex[pd_constants.FieldNames.PROMPT]
        + ex.get(pd_constants.FieldNames.TARGET, "")
        for ex in inputs
    ]
    preprocessed_texts = self.encode_inputs(texts)
    batched_outputs = {
        "token_ids": preprocessed_texts["token_ids"],
        "padding_mask": preprocessed_texts["padding_mask"],
    }
    # Convert to numpy for post-processing.
    detached_outputs = {
        k: keras.ops.convert_to_numpy(v) for k, v in batched_outputs.items()
    }
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = lit_utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def output_spec(self) -> lit_types.Spec:
    return pd_constants.OUTPUT_SPEC_TOKENIZER


def initialize_model_group_for_salience(
    new_name: str, **kw
) -> lit_model.ModelMap:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  salience_name, tokenizer_name = pd_utils.generate_model_group_names(new_name)
  generation_model = KerasGenerationModel(**kw)
  salience_model = KerasSalienceModel(model=generation_model.model, **kw)
  tokenizer_model = KerasTokenizerModel(model=generation_model.model, **kw)
  return {
      new_name: generation_model,
      salience_name: salience_model,
      tokenizer_name: tokenizer_model,
  }
