"""Wrapper for HuggingFace models in LIT.

Includes BERT masked LM, GPT-2, and T5.

This wrapper loads a model into memory and implements the a number of helper
functions to predict a batch of examples and extract information such as
hidden states and attention.
"""
from collections.abc import Sequence
import enum
import functools
import re

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
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


_DEFAULT_MAX_LENGTH = 1024
_PYTORCH = "torch"
_TENSORFLOW = "tensorflow"
# HuggingFace uses two letter abbreviations for pytorch and tensorflow.
_HF_PYTORCH = "pt"
_HF_TENSORFLOW = "tf"


class BertMLM(lit_model.BatchedModel):
  """BERT masked LM using Huggingface Transformers and TensorFlow 2."""

  MASK_TOKEN = "[MASK]"

  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  @classmethod
  def init_spec(cls) -> lit_model.Spec:
    return {
        "model_name_or_path": lit_types.String(default="bert-base-uncased"),
        "top_k": lit_types.Integer(default=10, min_val=1, max_val=25),
    }

  def __init__(self, model_name_or_path="bert-base-uncased", top_k=10):
    super().__init__()

    # Normally path is a directory; if it's an archive file, download and
    # extract to the transformers cache.
    if model_name_or_path.endswith(".tar.gz"):
      model_name_or_path = file_cache.cached_path(
          model_name_or_path, extract_compressed_file=True
      )

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False
    )
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    self.model = model_utils.load_pretrained(
        transformers.TFBertForMaskedLM,
        model_name_or_path,
        output_hidden_states=True,
        output_attentions=True,
    )
    self.top_k = top_k

  # TODO(lit-dev): break this out as a helper function, write some tests,
  # and de-duplicate code with the other text generation functions.
  def _get_topk_tokens(
      self, scores: np.ndarray
  ) -> list[list[tuple[str, float]]]:
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

  def _postprocess(self, output: dict[str, np.ndarray]):
    """Postprocess, modifying output dict in-place."""
    # Slice to remove padding, omitting initial [CLS] and final [SEP]
    slicer = slice(1, output.pop("ntok") - 1)
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(
        output.pop("input_ids")[slicer])
    probas = output.pop("probas")

    # Predictions at every position, regardless of masking.
    output["pred_tokens"] = self._get_topk_tokens(probas[slicer])  # pytype: disable=container-type-mismatch

    return output

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # If input has a 'tokens' field, use that. Otherwise tokenize the text.
    tokenized_texts = [
        ex.get("tokens") or self.tokenizer.tokenize(ex["text"]) for ex in inputs
    ]
    encoded_input = model_utils.batch_encode_pretokenized(
        self.tokenizer, tokenized_texts)

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

  def load(self, model_name_or_path):
    """Dynamically load a new BertMLM model given a model name."""
    return BertMLM(model_name_or_path, self.top_k)

  def input_spec(self):
    return {
        "text": lit_types.TextSegment(),
        "tokens": lit_types.Tokens(mask_token="[MASK]", required=False),
    }

  def output_spec(self):
    return {
        "tokens": lit_types.Tokens(parent="text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "cls_emb": lit_types.Embeddings(),
    }


# TODO(lit-dev): merge with below, inherit from HFBaseModel.
class GPT2LanguageModel(lit_model.BatchedModel):
  """Wrapper for a Huggingface Transformers GPT-2 model.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions plus additional helper functions to
  convert and clean tokens and to compute the top_k predictions from logits.
  """

  @property
  def num_layers(self):
    return self.model.config.n_layer

  @classmethod
  def init_spec(cls) -> lit_model.Spec:
    return {
        "model_name_or_path": lit_types.String(default="gpt2"),
        "top_k": lit_types.Integer(default=10, min_val=1, max_val=25),
    }

  def __init__(self, model_name_or_path="gpt2", top_k=10):
    """Constructor for GPT2LanguageModel.

    Args:
      model_name_or_path: gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2,
        etc.
      top_k: How many predictions to prune.
    """
    super().__init__()

    # Normally path is a directory; if it's an archive file, download and
    # extract to the transformers cache.
    if model_name_or_path.endswith(".tar.gz"):
      model_name_or_path = file_cache.cached_path(
          model_name_or_path, extract_compressed_file=True
      )

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False
    )
    # Set this after init, as if pad_token= is passed to
    # AutoTokenizer.from_pretrained() above it will create a new token with
    # with id = max_vocab_length and cause out-of-bounds errors in
    # the embedding lookup.
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model = transformers.TFGPT2LMHeadModel.from_pretrained(
        model_name_or_path, output_hidden_states=True, output_attentions=True
    )
    self.top_k = top_k

  @staticmethod
  def clean_bpe_token(tok):
    if not tok.startswith("Ġ"):
      return "_" + tok
    else:
      return tok.replace("Ġ", "")

  def ids_to_clean_tokens(self, ids):
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_bpe_token(t) for t in tokens]

  def _pred(self, encoded_inputs):
    """Predicts one batch of tokenized text.

    Also performs some batch-level post-processing in TF.
    Single-example postprocessing is done in _postprocess(), and operates on
    numpy arrays.

    Each prediction has the following returns:
    logits: tf.Tensor (batch_size, sequence_length, config.vocab_size).
    past: list[tf.Tensor] of length config.n_layers with each tensor shape
             (2, batch_size, num_heads, sequence_length, embed_size_per_head)).
    states: Tuple of tf.Tensor (one for embeddings + one for each layer),
            with shape (batch_size, sequence_length, hidden_size).
    attentions: Tuple of tf.Tensor (one for each layer) with shape
                (batch_size, num_heads, sequence_length, sequence_length)
    Within this function, we combine each Tuple/List into a single Tensor.

    Args:
      encoded_inputs: output of self.tokenizer()

    Returns:
      payload: Dictionary with items described above, each as single Tensor.
    """
    out: transformers.modeling_tf_outputs.TFCausalLMOutputWithPast = \
        self.model(encoded_inputs["input_ids"])

    model_probs = tf.nn.softmax(out.logits, axis=-1)
    top_k = tf.math.top_k(model_probs, k=self.top_k, sorted=True, name=None)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }

    # Convert representations for each layer from tuples to single Tensor.
    for i in range(len(out.attentions)):
      batched_outputs[f"layer_{i+1:d}_attention"] = out.attentions[i]
    for i in range(len(out.hidden_states)):
      batched_outputs[f"layer_{i:d}_avg_embedding"] = tf.math.reduce_mean(
          out.hidden_states[i], axis=1)

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    ntok = preds.pop("ntok")
    ids = preds.pop("input_ids")[:ntok]
    preds["tokens"] = self.ids_to_clean_tokens(ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a list[list[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop("top_k_probs")[:ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self.ids_to_clean_tokens(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    # Process attention.
    for key in preds:
      if not re.match(r"layer_(\d+)/attention", key):
        continue
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
    # The BatchedModel base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [ex["text"] for ex in inputs]
    encoded_inputs = self.tokenizer(
        texts,
        return_tensors="tf",
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )

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
      spec[f"layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
          align_in="tokens", align_out="tokens")
      spec[f"layer_{i:d}_avg_embedding"] = lit_types.Embeddings()
    return spec


@enum.unique
class MLFramework(enum.Enum):
  """The supported deep learning frameworks."""

  PT = _PYTORCH
  TF = _TENSORFLOW


class HFBaseModel(lit_model.BatchedModel):
  """Base class for HF generative, salience, tokenizer model wrappers."""

  # Enum str values for entries in MLFramework, used for init_spec and logging.
  _ML_FRAMEWORK_VALUES = [framework.value for framework in MLFramework]

  @property
  def num_layers(self):
    return self.model.config.n_layer

  @classmethod
  def init_spec(cls) -> lit_model.Spec:
    return {
        "model_name_or_path": lit_types.String(default="gpt2"),
        "batch_size": lit_types.Integer(default=6, min_val=1, max_val=64),
        "framework": lit_types.CategoryLabel(vocab=cls._ML_FRAMEWORK_VALUES),
    }

  def __init__(
      self,
      model_name_or_path="gpt2",
      batch_size=6,
      framework=_PYTORCH,
      model=None,
      tokenizer=None,
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
      if model_name_or_path.endswith(".tar.gz"):
        model_name_or_path = file_cache.cached_path(
            model_name_or_path, extract_compressed_file=True
        )

      # Note: we need to left-pad for generation to work properly.
      # Other modes such as scoring and salience should handle this as well;
      # see example in HFSalienceModel._postprocess().
      self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_name_or_path,
          use_fast=False,
          padding_side="left",
          model_max_length=_DEFAULT_MAX_LENGTH,
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
            f" supported, please choose from {self._ML_FRAMEWORK_VALUES}."
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
    return {
        "prompt": lit_types.TextSegment(),
        "target": lit_types.TextSegment(required=False),
    }


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
        "max_new_tokens": lit_types.Integer(default=50, min_val=1, max_val=500)
    }

  def __init__(self, *args, max_new_tokens=50, **kw):
    """Constructor for HFGenerativeModel.

    Args:
      *args: as to HFBaseModel.__init__
      max_new_tokens: the maximum number of new tokens to generate.
      **kw: as to HFBaseModel.__init__
    """
    super().__init__(*args, **kw)
    self.max_new_tokens = max_new_tokens

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # TODO(b/324957491): return actual decoder scores for each generation.
    # GeneratedTextCandidates should be a list[(text, score)]
    preds["response"] = [(preds["response"], 1.0)]
    ntok_in = preds.pop("ntok_in")
    embs = preds.pop("embs")
    # Mean-pool over input tokens.
    preds["prompt_embeddings"] = np.mean(
        embs[-(self.max_new_tokens + ntok_in) : -self.max_new_tokens], axis=0
    )
    # Mean-pool over output (generated) tokens.
    # TODO(b/324957491): slice this to only "real" output tokens,
    # if generation length < max generation length.
    preds["response_embeddings"] = np.mean(embs[-self.max_new_tokens :], axis=0)

    return preds

  ##
  # LIT API implementations
  def predict_minibatch(self, inputs):
    prompts = [ex["prompt"] for ex in inputs]
    encoded_inputs = self.tokenizer(
        prompts,
        return_tensors=_HF_PYTORCH
        if self.framework == MLFramework.PT
        else _HF_TENSORFLOW,
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )
    if self.framework == MLFramework.PT:
      encoded_inputs = encoded_inputs.to(self.device)

    outputs = self.model.generate(
        encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        max_new_tokens=self.max_new_tokens,
    )

    responses = self.tokenizer.batch_decode(
        outputs[:, -self.max_new_tokens :], skip_special_tokens=True
    )

    if self.framework == MLFramework.PT:
      with torch.no_grad():
        # Input embeddings: <float>[batch_size, num_tokens, emb_dim]
        embeddings = self.embedding_table(outputs)

      batched_outputs = {
          "embs": embeddings.cpu().to(torch.float),
          "ntok_in": (
              torch.sum(encoded_inputs["attention_mask"], axis=1)
              .cpu()
              .to(torch.int)
          ),
      }
    else:
      embeddings = self.embedding_table(outputs)
      batched_outputs = {
          "embs": embeddings,
          "ntok_in": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
          # TODO(b/324957491): compute ntok_out if < max_output_tokens ?
      }

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    detached_outputs["response"] = responses
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def output_spec(self) -> lit_types.Spec:
    return {
        "response": lit_types.GeneratedTextCandidates(parent="target"),
        "prompt_embeddings": lit_types.Embeddings(required=False),
        "response_embeddings": lit_types.Embeddings(required=False),
    }


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
        # Gradients are already aligned to input tokens.
        "grad_l2": grad_l2,
        "grad_dot_input": grad_dot_input,
        # Shift token loss to align with (input) tokens.
        # "token_loss": tf.roll(per_token_loss, shift=1, axis=1),
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
        # Gradients are already aligned to input tokens.
        "grad_l2": grad_l2.cpu().to(torch.float),
        "grad_dot_input": grad_dot_input.cpu().to(torch.float),
        # Shift token loss to align with (input) tokens.
        # "token_loss": torch.roll(per_token_loss, shifts=1, dims=1),
    }

    return batched_outputs

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Be sure to cast to bool, otherwise this will select integer positions 0, 1
    # rather than acting as a boolean mask.
    mask = preds.pop("attention_mask").astype(bool)
    ids = preds.pop("input_ids")[mask]
    preds["tokens"] = self.ids_to_clean_tokens(ids)
    for key in utils.find_spec_keys(self.output_spec(), lit_types.TokenScores):
      preds[key] = preds[key][mask]
    # First token (usually <s>) is not actually predicted, so return 0 for loss.
    # preds["token_loss"][0] = 0

    return preds

  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [ex["prompt"] + ex.get("target", "") for ex in inputs]
    encoded_inputs = self.tokenizer(
        texts,
        return_tensors=_HF_PYTORCH
        if self.framework == MLFramework.PT
        else _HF_TENSORFLOW,
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first",
    )
    target_masks = [ex.get("target_mask", []) for ex in inputs]

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
    return super().input_spec() | {
        "target_mask": lit_types.TokenScores(align="", required=False),
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        "tokens": lit_types.Tokens(parent=""),  # all tokens
        "grad_l2": lit_types.TokenScores(align="tokens"),
        "grad_dot_input": lit_types.TokenScores(align="tokens"),
        # "token_loss": lit_types.TokenScores(align="tokens"),
    }


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
    preds["tokens"] = self.ids_to_clean_tokens(ids)
    return preds

  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # Preprocess inputs.
    texts = [ex["prompt"] + ex.get("target", "") for ex in inputs]
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
    return {
        "tokens": lit_types.Tokens(parent=""),  # all tokens
    }


def initialize_model_group_for_salience(
    name, *args, max_new_tokens=512, **kw
) -> dict[str, lit_model.Model]:
  """Creates '{name}' and '_{name}_salience' and '_{name}_tokenizer'."""
  generation_model = HFGenerativeModel(
      *args, **kw, max_new_tokens=max_new_tokens
  )
  salience_model = HFSalienceModel.from_loaded(generation_model)
  tokenizer_model = HFTokenizerModel.from_loaded(generation_model)
  return {
      name: generation_model,
      f"_{name}_salience": salience_model,
      f"_{name}_tokenizer": tokenizer_model,
  }
