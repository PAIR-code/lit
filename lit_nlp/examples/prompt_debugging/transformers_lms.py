"""Wrapper for HuggingFace models in LIT.

Supported models include Gemma, GPT-2, Llama, Mistral, etc.

This wrapper loads a model into memory and implements the a number of helper
functions to predict a batch of examples and extract information such as
hidden states and attention.
"""
from collections.abc import Sequence
import enum
import functools
from typing import Any, Mapping

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.prompt_debugging import constants as pd_constants
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils
import numpy as np
import transformers

# pylint: disable=g-import-not-at-top
# pytype: disable=import-error
try:
  import tensorflow as tf
except (ModuleNotFoundError, ImportError):
  logging.warning("TensorFlow is not available.")

try:
  import torch
except (ModuleNotFoundError, ImportError):
  logging.warning("PyTorch is not available.")
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top


_PYTORCH = "torch"
_TENSORFLOW = "tensorflow"
# HuggingFace uses two letter abbreviations for pytorch and tensorflow.
_HF_PYTORCH = "pt"
_HF_TENSORFLOW = "tf"


@enum.unique
class MLFramework(enum.Enum):
  """The supported deep learning frameworks."""

  PT = _PYTORCH
  TF = _TENSORFLOW


SUPPORTED_ML_RUNTIMES = [framework.value for framework in MLFramework]


class HFBaseModel(lit_model.BatchedModel):
  """Base class for HF generative, salience, tokenizer model wrappers."""

  # Enum str values for entries in MLFramework, used for init_spec and logging.

  @property
  def num_layers(self):
    return self.model.config.n_layer

  @classmethod
  def init_spec(cls) -> lit_model.Spec:
    return {
        "model_name_or_path": lit_types.String(default="gpt2"),
        "batch_size": lit_types.Integer(default=6, min_val=1, max_val=64),
        "framework": lit_types.CategoryLabel(vocab=SUPPORTED_ML_RUNTIMES),
    }

  def __init__(
      self,
      model_name_or_path="gpt2",
      batch_size=6,
      framework=_PYTORCH,
      model=None,
      tokenizer=None,
      **unused_kw,
  ):
    """Constructor for HF base model wrappers.

    Note: args "model" and "tokenizer" take priority if both are specified.
    Otherwise, "model_name_or_path" is used to initialize the model and
    tokenizer.

    This class supports common HF transformer models such as GPT2, Llama,
    Mistral, etc.

    Args:
      model_name_or_path: gpt2, gpt2-medium, gpt2-large, distilgpt2,
        meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.
      batch_size: the number of items to process per `predict_minibatch` call.
      framework: the deep learning framework, only "tensorflow" and "torch"
        are supported.
      model: an initialized transformer model.
      tokenizer: an initialized tokenizer.
    """
    super().__init__()

    if model is not None and tokenizer is not None:
      self.model = model
      self.tokenizer = tokenizer
      # Check if the HF model object's framework is supported here.
      if model.framework == _HF_PYTORCH:
        self.framework = MLFramework.PT
      elif model.framework == _HF_TENSORFLOW:
        self.framework = MLFramework.TF
      else:
        raise ValueError(
            f"The HuggingFace model framework `{model.framework}` is not"
            " supported."
        )
    else:
      # Normally path is a directory; if it's an archive file, download and
      # extract to the transformers cache.
      if (
          is_tar_gz := model_name_or_path.endswith(".tar.gz")
      ) or file_cache.is_remote(model_name_or_path):
        model_name_or_path = file_cache.cached_path(
            model_name_or_path,
            extract_compressed_file=is_tar_gz,
        )

      # Note: we need to left-pad for generation to work properly.
      # Other modes such as scoring and salience should handle this as well;
      # see example in HFSalienceModel._postprocess().
      self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_name_or_path,
          use_fast=False,
          padding_side="left",
      )
      # Set this after init, as if pad_token= is passed to
      # AutoTokenizer.from_pretrained() above it will create a new token with
      # with id = max_vocab_length and cause out-of-bounds errors in
      # the embedding lookup.
      if framework == _PYTORCH:
        auto_model = transformers.AutoModelForCausalLM
        self.framework = MLFramework.PT
      elif framework == _TENSORFLOW:
        auto_model = transformers.TFAutoModelForCausalLM
        self.framework = MLFramework.TF
      else:
        raise ValueError(
            f"The provided value `{framework}` for arg `framework` is not"
            f" supported, please choose from {SUPPORTED_ML_RUNTIMES}."
        )
      self.model = auto_model.from_pretrained(
          model_name_or_path,
          output_hidden_states=True,
          output_attentions=False,
      )
    if self.framework == MLFramework.PT:
      self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
      self.model = self.model.to(self.device)
    self.embedding_table = self.model.get_input_embeddings()
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.batch_size = batch_size

  @property
  def pad_left(self):
    return self.tokenizer.padding_side == "left"

  @classmethod
  def from_loaded(cls, existing: "HFBaseModel", *args, **kw):
    """Share weights and underlying HF model with another instance."""
    return cls(model=existing.model, tokenizer=existing.tokenizer, *args, **kw)

  def clean_subword_token(self, tok):
    # For GPT2 tokenizer.
    tok = tok.replace("Ċ", "\n")  # newlines
    tok = tok.replace("Ġ", "▁")  # start of word -> magic underscore
    # For SentencePiece Tokenizer.
    tok = tok.replace("<0x0A>", "\n")  # newlines
    return tok

  def ids_to_clean_tokens(self, ids: Sequence[int]) -> list[str]:
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_subword_token(t) for t in tokens]

  def max_minibatch_size(self) -> int:
    # The BatchedModel base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return self.batch_size

  def input_spec(self):
    return pd_constants.INPUT_SPEC


class HFGenerativeModel(HFBaseModel):
  """Wrapper for a HF Transformer model that generates texts.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions to generate text responses given input
  prompts.

  Note that the default model generation config is used such that the response
  is produced using multinomial sampling.
  """

  @classmethod
  def init_spec(cls) -> lit_model.Spec:
    return super().init_spec() | {
        "max_length": lit_types.Integer(default=512, min_val=1, max_val=2048)
    }

  def __init__(self, *args, max_length=512, **kw):
    """Constructor for HFGenerativeModel.

    Args:
      *args: as to HFBaseModel.__init__
      max_length: the maximum length the generated tokens can have. Corresponds
        to the length of the input prompt + max_new_tokens.
      **kw: as to HFBaseModel.__init__
    """
    super().__init__(*args, **kw)
    self.max_length = max_length

  def _postprocess(self, preds: Mapping[str, Any]) -> Mapping[str, Any]:
    """Post-process single-example preds. Operates on numpy arrays.

    Args:
      preds: a dict of the model outputs, including the response text, number of
        input and output tokens, and the embeddings of the input and output
        tokens (merged into a single array).

    Returns:
      a dict of the processed model outputs, including the response texts and
        embeddings of the input and output tokens (separated into two arrays).
    """
    # TODO(b/324957491): return actual decoder scores for each generation. For
    # now, we only output GeneratedText.
    processed_preds = {}
    processed_preds[pd_constants.FieldNames.RESPONSE] = preds[
        pd_constants.FieldNames.RESPONSE
    ]
    ntok_in = preds["ntok_in"]
    ntok_out = preds["ntok_out"]
    embs = preds["embs"]
    assert embs.shape[0] >= ntok_in + ntok_out
    # Mean-pool over input tokens.
    processed_preds[pd_constants.FieldNames.PROMPT_EMBEDDINGS] = np.mean(
        embs[-(ntok_out + ntok_in) : -ntok_out], axis=0
    )
    # Mean-pool over output (generated) tokens.
    # TODO(b/324957491): slice this to only "real" output tokens,
    # if generation length < max generation length.
    processed_preds[pd_constants.FieldNames.RESPONSE_EMBEDDINGS] = np.mean(
        embs[-ntok_out:], axis=0
    )

    return processed_preds

  def _get_batched_outputs(
      self, inputs: Sequence[Mapping[str, lit_types.TextSegment]]
  ) -> Mapping[str, Any]:
    """Returns the batched outputs generated by the model for the given inputs.

    Args:
      inputs: model inputs containing text prompts.

    Returns:
      a dict of the model outputs, including the generated texts and auxiliary
        data in numpy arrays (could come from torch or tensorflow, depending on
        the transformer backend).
    """
    encoded_inputs = self.tokenizer(
        [ex["prompt"] for ex in inputs],
        return_tensors=(
            _HF_PYTORCH if self.framework == MLFramework.PT else _HF_TENSORFLOW
        ),
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )
    batch_size, ntok_in = encoded_inputs["input_ids"].shape

    if self.framework == MLFramework.PT:
      encoded_inputs = encoded_inputs.to(self.device)

    outputs = self.model.generate(**encoded_inputs, max_length=self.max_length)

    if isinstance(outputs, transformers.utils.ModelOutput):
      outputs = outputs.sequences

    ntok_out = outputs.shape[1] - ntok_in

    responses = self.tokenizer.batch_decode(
        outputs[:, -ntok_out:], skip_special_tokens=True
    )

    if self.framework == MLFramework.PT:
      with torch.no_grad():
        # Input embeddings: <float>[batch_size, num_tokens, emb_dim]
        embeddings = self.embedding_table(outputs).cpu().to(torch.float)
    else:
      embeddings = self.embedding_table(outputs)

    return {
        "embs": embeddings.numpy(),
        "ntok_in": np.array((batch_size, ntok_in)),
        "ntok_out": np.full((batch_size,), ntok_out),
        pd_constants.FieldNames.RESPONSE: responses,
    }

  ##
  # LIT API implementations
  def predict_minibatch(self, inputs):
    batched_outputs = self._get_batched_outputs(inputs)

    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(batched_outputs)
    return map(self._postprocess, unbatched_outputs)

  def output_spec(self) -> lit_types.Spec:
    return (
        pd_constants.OUTPUT_SPEC_GENERATION
        | pd_constants.OUTPUT_SPEC_GENERATION_EMBEDDINGS
    )


class HFSalienceModel(HFBaseModel):
  """Wrapper for a HF Transformer model that computes input (token) salience."""

  def _left_pad_target_masks(self, seq_length, target_masks):
    """Pads target masks (from left) to the desired sequence length.

    Args:
      seq_length: desired length of the padded masks.
      target_masks: list(array_like) of binary (0/1) masks for each input.

    Returns:
      Numpy array of the padded masks at the desired sequence length.
    """
    # It doesn't make sense to interpret the first token, since it is not ever
    # predicted. But we need to ensure that the mask[0] is zero, so it doesn't
    # cause problems when 'rolled' to the last position below.
    modified_masks = [[0] + list(mask[1:]) for mask in target_masks]
    pad_fn = functools.partial(
        utils.pad1d,
        min_len=seq_length,
        max_len=seq_length,
        pad_val=0,
        pad_left=self.pad_left,
    )
    padded_target_masks = np.stack(
        [pad_fn(mask) for mask in modified_masks],
        axis=0,
    )
    return padded_target_masks

  def _pred_tf(self, encoded_inputs, target_masks):
    """Predicts one batch of tokenized text using TF.

    Also performs some batch-level post-processing in TF.
    Single-example postprocessing is done in _postprocess(), and operates on
    numpy arrays.

    Args:
      encoded_inputs: output of self.tokenizer()
      target_masks: list(array_like) of binary (0/1) masks for each input

    Returns:
      payload: Dictionary with items described above, each as single Tensor.
    """
    input_ids = encoded_inputs["input_ids"]

    # <tf.int32>[batch_size, num_tokens]; ignore the last one in each row.
    target_ids = tf.roll(input_ids, shift=-1, axis=1)
    ##
    # Process target masks
    padded_target_masks = tf.constant(
        self._left_pad_target_masks(target_ids.shape[1], target_masks),
        dtype=tf.bool,
    )
    # Shift masks back so they align with target_ids.
    loss_mask = tf.roll(padded_target_masks, shift=-1, axis=1)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      # We need to run the embedding layer ourselves so we can trace it.
      # See here for how the model normally does this:
      # https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/gpt2/modeling_tf_gpt2.py#L450
      embs = self.embedding_table(input_ids)
      tape.watch(embs)

      out = self.model(
          input_ids=None,
          inputs_embeds=embs,
          attention_mask=encoded_inputs["attention_mask"],
      )

      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction="none"
      )
      # <tf.float>[batch_size, num_tokens]
      per_token_loss = loss_fn(target_ids, out.logits)
      masked_loss = per_token_loss * tf.cast(loss_mask, per_token_loss.dtype)

    grads = tape.gradient(
        masked_loss, embs
    )  # <tf.float>[batch_size, num_tokens, hdim]

    grad_l2 = tf.norm(grads, axis=2)  # <tf.float>[batch_size, num_tokens]
    grad_dot_input = tf.reduce_sum(
        grads * embs, axis=2
    )  # <tf.float>[batch_size, num_tokens]

    batched_outputs = {
        "input_ids": input_ids,
        "attention_mask": encoded_inputs["attention_mask"],
        pd_constants.FieldNames.GRAD_NORM: grad_l2,
        pd_constants.FieldNames.GRAD_DOT_INPUT: grad_dot_input,
    }

    return batched_outputs

  def _pred_pt(self, encoded_inputs, target_masks):
    """Predicts one batch of tokenized text using PyTorch.

    Also performs some batch-level post-processing in PyTorch.
    Single-example postprocessing is done in _postprocess(), and operates on
    numpy arrays.

    Args:
      encoded_inputs: output of self.tokenizer()
      target_masks: list(array_like) of binary (0/1) masks for each input

    Returns:
      payload: Dictionary with items described above, each as single Tensor.
    """
    encoded_inputs = encoded_inputs.to(self.device)
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # [batch_size, num_tokens]; ignore the last one in each row.
    target_ids = torch.roll(input_ids, shifts=-1, dims=1).to(self.device)
    ##
    # Process target masks
    padded_target_masks = torch.tensor(
        self._left_pad_target_masks(target_ids.shape[1], target_masks)
    ).bool()
    loss_mask = torch.roll(padded_target_masks, shifts=-1, dims=1).to(
        self.device
    )

    embs = self.embedding_table(input_ids)
    outs = self.model(
        input_ids=None,
        inputs_embeds=embs,
        attention_mask=attention_mask,
    )
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    # Need to reshape outs.logits from [batch_size, num_tokens, vocab_size]
    # to [batch_size, vocab_size, num_tokens] so the last dimension matches that
    # of target_ids with dimension [batch_size, num_tokens].
    per_token_loss = loss_func(outs.logits.permute(0, 2, 1), target_ids)
    masked_loss = per_token_loss * loss_mask

    # returned gradients are wrapped in a single item tuple.
    grads = torch.autograd.grad(
        masked_loss, embs, grad_outputs=torch.ones_like(masked_loss)
    )[0]

    # Remove the grad function from embs.
    embs = embs.detach()
    grad_l2 = torch.norm(grads, dim=2)  # [batch_size, num_tokens]
    grad_dot_input = torch.sum(grads * embs, axis=2)  # [batch_size, num_tokens]

    batched_outputs = {
        "input_ids": input_ids.cpu().to(torch.int),
        "attention_mask": attention_mask.cpu().to(torch.int),
        pd_constants.FieldNames.GRAD_NORM: grad_l2.cpu().to(torch.float),
        pd_constants.FieldNames.GRAD_DOT_INPUT: grad_dot_input.cpu().to(
            torch.float
        ),
    }

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Be sure to cast to bool, otherwise this will select integer positions 0, 1
    # rather than acting as a boolean mask.
    mask = preds.pop("attention_mask").astype(bool)
    ids = preds.pop("input_ids")[mask]
    preds[pd_constants.FieldNames.TOKENS] = self.ids_to_clean_tokens(ids)
    for key in utils.find_spec_keys(self.output_spec(), lit_types.TokenScores):
      preds[key] = preds[key][mask]
    # First token (usually <s>) is not actually predicted, so return 0 for loss.
    # preds["token_loss"][0] = 0

    return preds

  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [
        ex[pd_constants.FieldNames.PROMPT]
        + ex.get(pd_constants.FieldNames.TARGET, "")
        for ex in inputs
    ]
    encoded_inputs = self.tokenizer(
        texts,
        return_tensors=_HF_PYTORCH
        if self.framework == MLFramework.PT
        else _HF_TENSORFLOW,
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )
    target_masks = [
        ex.get(pd_constants.FieldNames.TARGET_MASK, []) for ex in inputs
    ]

    # Get the predictions.
    if self.framework == MLFramework.PT:
      batched_outputs = self._pred_pt(encoded_inputs, target_masks)
    else:
      batched_outputs = self._pred_tf(encoded_inputs, target_masks)

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self):
    return super().input_spec() | pd_constants.INPUT_SPEC_SALIENCE

  def output_spec(self) -> lit_types.Spec:
    return pd_constants.OUTPUT_SPEC_SALIENCE


class HFTokenizerModel(HFBaseModel):
  """Wrapper to run only the tokenizer.

  Should exactly match tokens from HFSalienceModel.
  """

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Be sure to cast to bool, otherwise this will select intger positions 0, 1
    # rather than acting as a boolean mask.
    mask = preds.pop("attention_mask").astype(bool)
    ids = preds.pop("input_ids")[mask]
    preds[pd_constants.FieldNames.TOKENS] = self.ids_to_clean_tokens(ids)
    return preds

  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [
        ex[pd_constants.FieldNames.PROMPT]
        + ex.get(pd_constants.FieldNames.TARGET, "")
        for ex in inputs
    ]
    encoded_inputs = self.tokenizer(
        texts,
        return_tensors=_HF_PYTORCH
        if self.framework == MLFramework.PT
        else _HF_TENSORFLOW,
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
    }
    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def output_spec(self) -> lit_types.Spec:
    return pd_constants.OUTPUT_SPEC_TOKENIZER


def initialize_model_group_for_salience(
    new_name: str, **kw
) -> lit_model.ModelMap:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  max_length = kw.pop("max_length", 512)
  salience_name, tokenizer_name = pd_utils.generate_model_group_names(new_name)
  generation_model = HFGenerativeModel(max_length=max_length, **kw)
  salience_model = HFSalienceModel.from_loaded(generation_model)
  tokenizer_model = HFTokenizerModel.from_loaded(generation_model)
  return {
      new_name: generation_model,
      salience_name: salience_model,
      tokenizer_name: tokenizer_model,
  }
