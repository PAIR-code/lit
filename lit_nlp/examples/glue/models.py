"""Wrapper for fine-tuned HuggingFace models in LIT."""

# TODO(b/261736863): Update to PEP 585 typings, consider using f-strings, and
# make common substrings into module CONSTANTS.

from collections.abc import Iterable, Sequence
import os
import re
import threading
from typing import Any, Callable, Optional

from absl import logging
import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.glue import model_utils
from lit_nlp.lib import file_cache
from lit_nlp.lib import utils
import numpy as np
import torch
import transformers

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec
_MF = transformers.modeling_outputs
SequenceClassifierOutput = _MF.SequenceClassifierOutput


@attr.s(auto_attribs=True, kw_only=True)
class GlueModelConfig(object):
  """Config options for a GlueModel."""

  # Preprocessing options
  max_seq_length: int = 128
  inference_batch_size: int = 32
  # Input options
  text_a_name: str = "sentence1"
  text_b_name: Optional[str] = "sentence2"  # set to None for single-segment
  label_name: str = "label"
  # Output options
  labels: Optional[list[str]] = None  # set to None for regression
  null_label_idx: Optional[int] = None
  compute_grads: bool = True  # if True, compute and return gradients.
  output_attention: bool = True
  output_embeddings: bool = True

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        "model_name_or_path": lit_types.String(
            default="bert-base-uncased",
            required=False,
        ),
        "max_seq_length": lit_types.Integer(
            default=128,
            max_val=512,
            min_val=1,
            required=False,
        ),
        "inference_batch_size": lit_types.Integer(
            default=32,
            max_val=64,
            min_val=1,
            required=False,
        ),
        "compute_grads": lit_types.Boolean(default=True, required=False),
        "output_attention": lit_types.Boolean(default=True, required=False),
        "output_embeddings": lit_types.Boolean(default=True, required=False),
    }


class GlueModel(lit_model.BatchedModel):
  """GLUE benchmark model, using PyTorch and Huggingface Transformers.

  This is a general-purpose classification or regression model. It works for
  one- or two-segment input, and predicts either a multiclass label or
  a regression score. See GlueModelConfig for available options.

  This implements the LIT API for inference (e.g. input_spec(), output_spec(),
  and predict()), but also provides a train() method to run fine-tuning.

  This is a full-featured implementation, which includes embeddings, attention,
  gradients, as well as support for the different input and output types above.
  """

  def _verify_num_layers(self, hidden_states: Sequence[Any]):
    """Verify correct # of layer activations returned."""
    # First entry is embeddings, then output from each transformer layer.
    expected_hidden_states_len = self.model.config.num_hidden_layers + 1
    actual_hidden_states_len = len(hidden_states)
    if actual_hidden_states_len != expected_hidden_states_len:
      raise ValueError(
          "Unexpected size of hidden_states. Should be one "
          "more than the number of hidden layers to account "
          "for the embeddings. Expected "
          f"{expected_hidden_states_len}, got "
          f"{actual_hidden_states_len}."
      )

  @property
  def is_regression(self) -> bool:
    return self.config.labels is None

  # TODO(b/254110131): Move file_cache.cached_path() call inside this __init__
  # function to reduce boilerplate in other locations (e.g., TCAV tests).
  def __init__(self, model_name_or_path="bert-base-uncased", **config_kw):
    self.config = GlueModelConfig(**config_kw)
    self._load_model(model_name_or_path)
    self._lock = threading.Lock()

  def _load_model(self, model_name_or_path):
    """Load model. Can be overridden for testing."""
    # Normally path is a directory; if it's an archive file, download and
    # extract to the transformers cache.
    if model_name_or_path.endswith(".tar.gz"):
      model_name_or_path = file_cache.cached_path(
          model_name_or_path, extract_compressed_file=True
      )

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path
    )
    self.vocab = self.tokenizer.convert_ids_to_tokens(
        range(len(self.tokenizer))
    )
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=1 if self.is_regression else len(self.config.labels),
        output_attentions=self.config.output_attention,
    )
    cls = transformers.AutoModelForSequenceClassification
    self.model = cls.from_pretrained(model_name_or_path, config=model_config)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.model.eval()

  def _get_tokens(self, ex: JsonDict, field_name: str) -> list[str]:
    with self._lock:
      return ex.get("tokens_" + field_name) or self.tokenizer.tokenize(
          ex[field_name]
      )

  def _preprocess(
      self, inputs: Iterable[JsonDict]
  ) -> transformers.BatchEncoding:
    # Use pretokenized input if available.
    tokens_a = [self._get_tokens(ex, self.config.text_a_name) for ex in inputs]
    tokens_b = None
    if self.config.text_b_name:
      tokens_b = [
          self._get_tokens(ex, self.config.text_b_name) for ex in inputs
      ]
    # Use custom tokenizer call to make sure we don't mangle pre-split
    # wordpieces in pretokenized input.
    encoded_input = model_utils.batch_encode_pretokenized(
        self.tokenizer,
        tokens_a,
        tokens_b,
        max_length=self.config.max_seq_length,
    )
    encoded_input.to(self.device)
    return encoded_input

  def _make_label_tensor(self, inputs: Iterable[JsonDict]) -> torch.Tensor:
    """Make a label tensor from inputs in LIT format."""
    if self.is_regression:
      return torch.tensor(
          [ex[self.config.label_name] for ex in inputs],
          dtype=torch.float32,
          device=self.device,
      )
    else:
      indexes = []
      if self.config.labels is not None:
        for ex in inputs:
          indexes.append(self.config.labels.index(ex[self.config.label_name]))
      return torch.tensor(
          indexes, dtype=torch.long, device=self.device
      )

  def train(
      self,
      train_inputs: list[JsonDict],
      validation_inputs: list[JsonDict],
      learning_rate=2e-5,
      batch_size=32,
      num_epochs=3,
      on_epoch_end: Optional[Callable[[int, dict[str, float]], None]] = None,
  ):
    """Run fine-tuning.

    Args:
      train_inputs: list of training examples in LIT format.
      validation_inputs: list of validation examples in LIT format.
      learning_rate: learning rate for the Adam optimizer.
      batch_size: training batch size.
      num_epochs: number of epochs to train for.
      on_epoch_end: optional callback, called after each epoch with
        (epoch_index, logs) where logs contains epoch metrics.

    Returns:
      Dict summarizing training history, with keys "epochs", "history",
      "params", and "optimizer_params".
    """
    self.model.train()
    optimizer = torch.optim.Adam(
        self.model.parameters(), lr=learning_rate, eps=1e-08
    )
    if self.is_regression:
      loss_fn = torch.nn.MSELoss()
    else:
      loss_fn = torch.nn.CrossEntropyLoss()

    # Use larger batch for validation since inference is about 1/2 memory usage
    # of backprop.
    eval_batch_size = 2 * batch_size
    steps_per_epoch = len(train_inputs) // batch_size
    history: dict[str, list[float]] = {
        "loss": [],
        "val_loss": [],
        ("val_rmse" if self.is_regression else "val_accuracy"): [],
    }
    for epoch in range(num_epochs):
      self.model.train()
      perm = torch.randperm(len(train_inputs))
      train_inputs = [train_inputs[i] for i in perm.tolist()]
      epoch_losses = []
      for step in range(steps_per_epoch):
        batch = train_inputs[step * batch_size : (step + 1) * batch_size]
        encoded_input = self._preprocess(batch)
        labels = self._make_label_tensor(batch)
        optimizer.zero_grad()
        out: SequenceClassifierOutput = self.model(
            **encoded_input, labels=labels
        )
        loss = out.loss
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

      val_loss, val_metric = self._evaluate(
          validation_inputs, eval_batch_size, loss_fn
      )
      history["loss"].append(float(np.mean(epoch_losses)))
      history["val_loss"].append(val_loss)
      history[
          ("val_rmse" if self.is_regression else "val_accuracy")
      ].append(val_metric)
      logs = {k: v[-1] for k, v in history.items()}
      logging.info(
          "Epoch %d: %s",
          epoch,
          ", ".join(f"{k}={v:.4f}" for k, v in logs.items()),
      )
      if on_epoch_end is not None:
        on_epoch_end(epoch, logs)

    self.model.eval()
    return {
        "epochs": list(range(num_epochs)),
        "history": history,
        "params": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
        },
        "optimizer_params": {
            "learning_rate": learning_rate,
            "epsilon": 1e-08,
        },
    }

  def _evaluate(
      self,
      eval_inputs: list[JsonDict],
      batch_size: int,
      loss_fn: torch.nn.Module,
  ) -> tuple[float, float]:
    """Run a validation pass, returning (loss, rmse_or_accuracy)."""
    self.model.eval()
    losses, sq_errors, correct, total = [], 0.0, 0, 0
    with torch.no_grad():
      for start in range(0, len(eval_inputs), batch_size):
        batch = eval_inputs[start : start + batch_size]
        encoded_input = self._preprocess(batch)
        labels = self._make_label_tensor(batch)
        out: SequenceClassifierOutput = self.model(
            **encoded_input, labels=labels
        )
        losses.append(out.loss.item())
        if self.is_regression:
          sq_errors += float(((out.logits[:, 0] - labels) ** 2).sum())
          total += len(batch)
        else:
          correct += int(
              (out.logits.argmax(dim=-1) == labels).sum()
          )
          total += len(batch)
    val_loss = float(np.mean(losses)) if losses else float("nan")
    if self.is_regression:
      val_metric = float(np.sqrt(sq_errors / max(total, 1)))
    else:
      val_metric = correct / max(total, 1)
    return val_loss, val_metric

  def save(self, path: str):
    """Save model weights and tokenizer info.

    To re-load, pass the path to the constructor instead of the name of a
    base model.

    Args:
      path: directory to save to. Will write several files here.
    """
    if not os.path.isdir(path):
      os.mkdir(path)
    self.tokenizer.save_pretrained(path)
    self.model.save_pretrained(path)

  def _segment_slicers(self, tokens: list[str]):
    """Slicers along the tokens dimension for each segment.

    For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'],
    we want to get the slices [a0, a1, ...] and [b0, b1, ...]

    Args:
      tokens: <string>[num_tokens], including special tokens

    Returns:
      (slicer_a, slicer_b), slice objects
    """
    try:
      split_point = tokens.index(self.tokenizer.sep_token)
    except ValueError:
      split_point = len(tokens) - 1
    slicer_a = slice(1, split_point)  # start after [CLS]
    slicer_b = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
    return slicer_a, slicer_b

  def _postprocess(self, output: dict[str, Any]):
    """Per-example postprocessing, on NumPy output."""
    ntok = output.pop("ntok")
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(
        output.pop("input_ids")[:ntok]
    )

    # Tokens for each segment, individually.
    slicer_a, slicer_b = self._segment_slicers(output["tokens"])
    output["tokens_" + self.config.text_a_name] = output["tokens"][slicer_a]
    if self.config.text_b_name:
      output["tokens_" + self.config.text_b_name] = output["tokens"][slicer_b]

    # Embeddings for each segment, individually.
    if self.config.output_embeddings:
      output["input_embs_" + self.config.text_a_name] = output["input_embs"][
          slicer_a
      ]
      if self.config.text_b_name:
        output["input_embs_" + self.config.text_b_name] = output["input_embs"][
            slicer_b
        ]

    # Gradients for each segment, individually.
    if self.config.compute_grads:
      # Gradients for the CLS token.
      output["cls_grad"] = output["input_emb_grad"][0]
      output["token_grad_" + self.config.text_a_name] = output[
          "input_emb_grad"
      ][slicer_a]
      if self.config.text_b_name:
        output["token_grad_" + self.config.text_b_name] = output[
            "input_emb_grad"
        ][slicer_b]

      # TODO(b/294613507): remove output[self.config.label_name] once TCAV
      # is updated.
      if not self.is_regression:
        # Return the label corresponding to the class index used for gradients.
        output[self.config.label_name] = self.config.labels[
            output[self.config.label_name]
        ]  # pytype: disable=container-type-mismatch

      # Remove "input_emb_grad" since it's not in the output spec.
      del output["input_emb_grad"]

    if not self.config.output_attention:
      return output

    # Process attention.
    for key in output:
      if not re.match(r"layer_(\d+)/attention", key):
        continue
      # Select only real tokens, since most of this matrix is padding.
      # <float32>[num_heads, max_seq_length, max_seq_length]
      # -> <float32>[num_heads, num_tokens, num_tokens]
      output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
      # Make a copy of this array to avoid memory leaks, since NumPy otherwise
      # keeps a pointer around that prevents the source array from being GCed.
      output[key] = output[key].copy()  # pytype: disable=attribute-error

    return output

  def _scatter_embs(
      self, passed_input_embs, input_embs, batch_indices, offsets
  ):
    """Scatters custom passed embeddings into the default model embeddings.

    Args:
      passed_input_embs: <float32>[num_scatter_tokens], the custom passed
        embeddings to be scattered into the default model embeddings.
      input_embs: the default model embeddings, as a torch.Tensor or ndarray.
      batch_indices: the indices of the embeddings to replace in the format
        (batch_index, sequence_index).
      offsets: the offset from which to scatter the custom embedding (number of
        tokens from the start of the sequence).

    Returns:
      The default model embeddings with scattered custom embeddings.
    """

    # <float32>[scatter_batch_size, num_tokens, emb_size]
    filtered_embs = [emb for emb in passed_input_embs if emb is not None]

    # Prepares update values that should be scattered in, i.e. one for each
    # of the (scatter_batch_size * num_tokens) word embeddings.
    # <np.float32>[scatter_batch_size * num_tokens, emb_size]
    updates = np.concatenate(filtered_embs)

    # Prepares indices in format (batch_index, sequence_index) for all
    # values that should be scattered in, i.e. one for each of the
    # (scatter_batch_size * num_tokens) word embeddings.
    scatter_indices = []
    for batch_index, sentence_embs, offset in zip(
        batch_indices, filtered_embs, offsets
    ):
      for token_index, _ in enumerate(sentence_embs):
        scatter_indices.append([batch_index, token_index + offset])

    # Scatters passed word embeddings into embeddings gathered from tokens.
    # <float32>[batch_size, num_tokens + num_special_tokens, emb_size]
    if not isinstance(input_embs, torch.Tensor):
      input_embs = torch.from_numpy(np.asarray(input_embs))
    indices = torch.tensor(scatter_indices, dtype=torch.long).t()
    updates_tensor = (
        torch.from_numpy(np.asarray(updates))
        .to(device=input_embs.device, dtype=input_embs.dtype)
    )
    return input_embs.index_put(
        (indices[0], indices[1]), updates_tensor
    )

  def scatter_all_embeddings(self, inputs, input_embs):
    """Scatters custom passed embeddings for text segment inputs.

    Args:
      inputs: the model inputs, which contain any custom embeddings to scatter.
      input_embs: the default model embeddings.

    Returns:
      The default model embeddings with scattered custom embeddings.
    """
    # Gets batch indices of any word embeddings that were passed for text_a.
    passed_input_embs_a = [
        ex.get("input_embs_" + self.config.text_a_name) for ex in inputs
    ]
    batch_indices_a = [
        index
        for (index, emb) in enumerate(passed_input_embs_a)
        if emb is not None
    ]

    # If word embeddings were passed in for text_a, scatter them into the
    # embeddings, gathered from the input ids. 1 is passed in as the offset
    # for each, since text_a starts at index 1, after the [CLS] token.
    if batch_indices_a:
      input_embs = self._scatter_embs(
          passed_input_embs_a,
          input_embs,
          batch_indices_a,
          offsets=np.ones(len(batch_indices_a), dtype=np.int64),
      )

    if self.config.text_b_name:
      # Gets batch indices of any word embeddings that were passed for text_b.
      passed_input_embs_b = [
          ex.get("input_embs_" + self.config.text_b_name) for ex in inputs
      ]
      batch_indices_b = [
          index
          for (index, emb) in enumerate(passed_input_embs_b)
          if emb is not None
      ]

      # If word embeddings were also passed in for text_b, scatter them into the
      # embeddings gathered from the input ids. The offsets are the [lengths
      # of the corresponding text_a embeddings] + 2, since text_b starts after
      # [CLS] [text_a tokens] [SEP]. (This assumes that text_b embeddings
      # will only be passed together with text_a embeddings.)
      if batch_indices_b:
        lengths = np.array(
            [len(embed) for embed in passed_input_embs_a if embed is not None]
        )
        input_embs = self._scatter_embs(
            passed_input_embs_b,
            input_embs,
            batch_indices_b,
            offsets=(lengths + 2),
        )
    return input_embs

  def get_target_scores(self, inputs: Iterable[JsonDict], scores):
    """Get target-class scores, as a 1D tensor.

    Args:
      inputs: list of input examples
      scores: <float32>[batch_size, num_classes], either logits or probas

    Returns:
      <float32>[batch_size] target scores for each input
    """
    arg_max = scores.argmax(dim=-1).cpu().numpy()
    grad_classes = [
        ex.get(self.config.label_name, arg_max[i])
        for (i, ex) in enumerate(inputs)
    ]
    # Convert the class names to indices if needed.
    grad_idxs = []
    for label in grad_classes:
      if isinstance(label, str) and self.config.labels is not None:
        grad_idxs.append(self.config.labels.index(label))
      else:
        grad_idxs.append(label)
    # <float32>[batch_size]
    batch_idx = torch.arange(len(inputs), device=scores.device)
    idx_tensor = torch.tensor(
        grad_idxs, dtype=torch.long, device=scores.device
    )
    return scores[batch_idx, idx_tensor], grad_idxs

  ##
  # LIT API implementation
  def max_minibatch_size(self):
    return self.config.inference_batch_size

  def get_embedding_table(self):
    return (
        self.vocab,
        self.model.get_input_embeddings().weight.detach().cpu().numpy(),
    )

  def predict_minibatch(self, inputs: Iterable[JsonDict]):
    encoded_input = self._preprocess(inputs)

    grad_context = (
        torch.enable_grad()
        if self.config.compute_grads
        else torch.no_grad()
    )
    with grad_context:
      # Gathers word embeddings from the model embedding layer using input ids
      # of the tokens.
      input_ids = encoded_input["input_ids"]
      word_embeddings = self.model.get_input_embeddings().weight
      # <float32>[batch_size, num_tokens, emb_size]
      input_embs = torch.nn.functional.embedding(input_ids, word_embeddings)

      # Scatter in any passed in embeddings.
      # <float32>[batch_size, num_tokens, emb_size]
      input_embs = self.scatter_all_embeddings(inputs, input_embs)

      model_inputs = {
          k: v for k, v in encoded_input.items() if k != "input_ids"
      }
      out: SequenceClassifierOutput = self.model(
          **model_inputs,
          inputs_embeds=input_embs,
          output_hidden_states=True,
          output_attentions=True,
      )

      batched_outputs = {
          "input_ids": encoded_input["input_ids"],
          "ntok": encoded_input["attention_mask"].sum(dim=1),
          "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
      }

      if self.config.output_embeddings:
        batched_outputs["input_embs"] = input_embs

        self._verify_num_layers(out.hidden_states)

        # <float32>[batch_size, num_tokens, 1]
        token_mask = torch.unsqueeze(
            encoded_input["attention_mask"].to(torch.float32), dim=2
        )
        # <float32>[batch_size, 1]
        denom = token_mask.sum(dim=1)
        for i, layer_output in enumerate(out.hidden_states):
          # layer_output is <float32>[batch_size, num_tokens, emb_dim]
          # average over tokens to get <float32>[batch_size, emb_dim]
          batched_outputs[f"layer_{i}/avg_emb"] = (
              layer_output * token_mask
          ).sum(dim=1) / denom

      if self.config.output_attention:
        if len(out.attentions) != self.model.config.num_hidden_layers:
          raise ValueError(
              "Unexpected size of attentions. Should be the same "
              "size as the number of hidden layers. Expected "
              f"{self.model.config.num_hidden_layers}, got "
              f"{len(out.attentions)}."
          )
        for i, layer_attention in enumerate(out.attentions):
          # <float32>[batch_size, num_heads, num_tokens, num_tokens]
          batched_outputs[f"layer_{i+1}/attention"] = layer_attention

      if self.is_regression:
        # <float32>[batch_size]
        batched_outputs["score"] = torch.squeeze(out.logits, dim=-1)
        # <float32>[batch_size], a single target per example
        scalar_targets = batched_outputs["score"]
      else:
        # <float32>[batch_size, num_labels]
        batched_outputs["probas"] = torch.softmax(out.logits, dim=-1)
        # <float32>[batch_size], a single target per example
        scalar_targets, grad_idxs = self.get_target_scores(
            inputs, batched_outputs["probas"]
        )
        # TODO(b/294613507): remove once TCAV updated.
        if self.config.compute_grads:
          batched_outputs[self.config.label_name] = torch.tensor(
              grad_idxs, device=self.device
          )

    # Request gradients after the forward pass is run.
    # Note: embs[0] includes position and segment encodings, as well as subword
    # embeddings.
    if self.config.compute_grads:
      # <float32>[batch_size, num_tokens, emb_dim]
      batched_outputs["input_emb_grad"] = torch.autograd.grad(
          scalar_targets.sum(), input_embs
      )[0]

    detached_outputs = {
        k: v.detach().cpu().numpy()
        for k, v in batched_outputs.items()
        if v is not None
    }
    # Sequence of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return map(self._postprocess, unbatched_outputs)

  def input_spec(self) -> Spec:
    ret = {}
    ret[self.config.text_a_name] = lit_types.TextSegment()
    ret["tokens_" + self.config.text_a_name] = lit_types.Tokens(
        parent=self.config.text_a_name, required=False
    )

    if self.config.text_b_name:
      ret[self.config.text_b_name] = lit_types.TextSegment()
      ret["tokens_" + self.config.text_b_name] = lit_types.Tokens(
          parent=self.config.text_b_name, required=False
      )

    if self.is_regression:
      ret[self.config.label_name] = lit_types.Scalar(required=False)
    else:
      ret[self.config.label_name] = lit_types.CategoryLabel(
          required=False, vocab=self.config.labels
      )

    if self.config.output_embeddings:
      # The input_embs_ fields are used for Integrated Gradients.
      text_a_embs = "input_embs_" + self.config.text_a_name
      ret[text_a_embs] = lit_types.TokenEmbeddings(
          align="tokens", required=False
      )
      if self.config.text_b_name:
        text_b_embs = "input_embs_" + self.config.text_b_name
        ret[text_b_embs] = lit_types.TokenEmbeddings(
            align="tokens", required=False
        )
    return ret

  def output_spec(self) -> Spec:
    ret = {"tokens": lit_types.Tokens()}
    ret["tokens_" + self.config.text_a_name] = lit_types.Tokens(
        parent=self.config.text_a_name
    )
    if self.config.text_b_name:
      ret["tokens_" + self.config.text_b_name] = lit_types.Tokens(
          parent=self.config.text_b_name
      )
    if self.is_regression:
      ret["score"] = lit_types.RegressionScore(parent=self.config.label_name)
    else:
      ret["probas"] = lit_types.MulticlassPreds(
          parent=self.config.label_name,
          vocab=self.config.labels,
          null_idx=self.config.null_label_idx,
      )

    if self.config.output_embeddings:
      ret["cls_emb"] = lit_types.Embeddings()
      # Average embeddings, one per layer including embeddings.
      for i in range(1 + self.model.config.num_hidden_layers):
        ret[f"layer_{i}/avg_emb"] = lit_types.Embeddings()

      # The input_embs_ fields are used for Integrated Gradients.
      ret["input_embs_" + self.config.text_a_name] = lit_types.TokenEmbeddings(
          align="tokens_" + self.config.text_a_name
      )
      if self.config.text_b_name:
        text_b_embs = "input_embs_" + self.config.text_b_name
        ret[text_b_embs] = lit_types.TokenEmbeddings(
            align="tokens_" + self.config.text_b_name
        )

    # Gradients, if requested.
    if self.config.compute_grads:
      ret["cls_grad"] = lit_types.Gradients(
          align=("score" if self.is_regression else "probas"),
          grad_for="cls_emb",
          grad_target_field_key=self.config.label_name,
      )
      if not self.is_regression:
        ret[self.config.label_name] = lit_types.CategoryLabel(
            required=False, vocab=self.config.labels
        )
      if self.config.output_embeddings:
        text_a_token_grads = "token_grad_" + self.config.text_a_name
        ret[text_a_token_grads] = lit_types.TokenGradients(
            align="tokens_" + self.config.text_a_name,
            grad_for="input_embs_" + self.config.text_a_name,
            grad_target_field_key=self.config.label_name,
        )
        if self.config.text_b_name:
          text_b_token_grads = "token_grad_" + self.config.text_b_name
          ret[text_b_token_grads] = lit_types.TokenGradients(
              align="tokens_" + self.config.text_b_name,
              grad_for="input_embs_" + self.config.text_b_name,
              grad_target_field_key=self.config.label_name,
          )

    if self.config.output_attention:
      # Attention heads, one field for each layer.
      for i in range(self.model.config.num_hidden_layers):
        ret[f"layer_{i+1}/attention"] = lit_types.AttentionHeads(
            align_in="tokens", align_out="tokens"
        )
    return ret


class SST2Model(GlueModel):
  """Classification model on SST-2."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name="sentence",
        text_b_name=None,
        labels=["0", "1"],
        null_label_idx=0,
        **kw,
    )


class MNLIModel(GlueModel):
  """Classification model on MultiNLI."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name="premise",
        text_b_name="hypothesis",
        labels=["entailment", "neutral", "contradiction"],
        **kw,
    )


class STSBModel(GlueModel):
  """Regression model on STS-B."""

  def __init__(self, *args, **kw):
    super().__init__(
        *args,
        text_a_name="sentence1",
        text_b_name="sentence2",
        labels=None,
        **kw,
    )

  def input_spec(self):
    ret = super().input_spec()
    ret[self.config.label_name] = lit_types.Scalar(min_val=0, max_val=5)
    return ret
