# Lint as: python3
r"""Code example for a custom model, using PyTorch.

This demo shows how to use a custom model with LIT, in just a few lines of code.
We'll use a transformers model, with a minimal amount of code to implement the
LIT API. Compared to models/glue_models.py, this has fewer features, but the
code is more readable.
This demo is similar in functionality to simple_tf2_demo.py, but uses PyTorch
instead of TensorFlow 2.
The transformers library can load weights from either,
so you can use any saved model compatible with the underlying model class
(AutoModelForSequenceClassification). To train something for this demo, you can:
- Use quickstart_sst_demo.py, and set --model_path to somewhere durable
- Or: Use tools/glue_trainer.py
- Or: Use any fine-tuning code that works with transformers, such as
https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts
To run locally:
  python -m lit_nlp.examples.simple_pytorch_demo \
      --port=5432 --model_path=/path/to/saved/model
Then navigate to localhost:5432 to access the demo UI.
NOTE: this demo still uses TensorFlow Datasets (which depends on TensorFlow) to
load the data. However, the output of glue.SST2Data is just NumPy arrays and
plain Python data, and you can easily replace this with a different library or
directly loading from CSV.
"""
import re
import sys

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.datasets import glue
from lit_nlp.lib import utils
import torch
import transformers

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

flags.DEFINE_string(
    "model_path", None,
    "Path to trained model, in standard transformers format, e.g. as "
    "saved by model.save_pretrained() and tokenizer.save_pretrained()")


def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from TensorFlow checkpoint (from_tf=True)")
    return cls.from_pretrained(*args, from_tf=True, **kw)


class SimpleSentimentModel(lit_model.Model):
  """Simple sentiment analysis model."""

  LABELS = ["0", "1"]  # negative, positive
  compute_grads: bool = True  # if True, compute and return gradients.

  def __init__(self, model_name_or_path):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        output_hidden_states=True,
        output_attentions=True,
    )
    # This is a just a regular PyTorch model.
    self.model = _from_pretrained(
        transformers.AutoModelForSequenceClassification,
        model_name_or_path,
        config=model_config)
    self.model.eval()

  ##
  # LIT API implementation
  def max_minibatch_size(self):
    # This tells lit_model.Model.predict() how to batch inputs to
    # predict_minibatch().
    # Alternately, you can just override predict() and handle batching yourself.
    return 32

  def predict_minibatch(self, inputs):

    # Preprocess to ids and masks, and make the input batch.
    encoded_input = self.tokenizer.batch_encode_plus(
        [ex["sentence"] for ex in inputs],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=128,
        padding="longest",
        truncation="longest_first")

    # Check and send to cuda (GPU) if available
    if torch.cuda.is_available():
      self.model.cuda()
      for tensor in encoded_input:
        encoded_input[tensor] = encoded_input[tensor].cuda()

    # Run a forward pass.
    with torch.set_grad_enabled(self.compute_grads):
      out: transformers.modeling_outputs.SequenceClassifierOutput = \
          self.model(**encoded_input)

    # Post-process outputs.
    batched_outputs = {
        "probas": torch.nn.functional.softmax(out.logits, dim=-1),
        "input_ids": encoded_input["input_ids"],
        "ntok": torch.sum(encoded_input["attention_mask"], dim=1),
        "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
    }

    # Add attention layers to batched_outputs
    assert len(out.attentions) == self.model.config.num_hidden_layers
    for i, layer_attention in enumerate(out.attentions):
      batched_outputs[f"layer_{i}/attention"] = layer_attention

    # Request gradients after the forward pass.
    # Note: hidden_states[0] includes position and segment encodings, as well as
    # subword embeddings.
    if self.compute_grads:
      # <torch.float32>[batch_size, num_tokens, emb_dim]
      scalar_pred_for_gradients = torch.max(
          batched_outputs["probas"], dim=1, keepdim=False, out=None)[0]
      batched_outputs["input_emb_grad"] = torch.autograd.grad(
          scalar_pred_for_gradients,
          out.hidden_states[0],
          grad_outputs=torch.ones_like(scalar_pred_for_gradients))[0]

    # Post-process outputs.
    # Return as NumPy for further processing.
    detached_outputs = {
        k: v.cpu().detach().numpy() for k, v in batched_outputs.items()}

    # Unbatch outputs so we get one record per input example.
    for output in utils.unbatch_preds(detached_outputs):
      ntok = output.pop("ntok")
      output["tokens"] = self.tokenizer.convert_ids_to_tokens(
          output.pop("input_ids")[:ntok])

      # set token gradients
      if self.compute_grads:
        output["token_grad_sentence"] = output["input_emb_grad"][:ntok]

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
        output[key] = output[key].copy()
      yield output

  def input_spec(self) -> lit_types.Spec:
    return {
        "sentence": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS, required=False)
    }

  def output_spec(self) -> lit_types.Spec:
    ret = {
        "tokens": lit_types.Tokens(),
        "probas": lit_types.MulticlassPreds(parent="label", vocab=self.LABELS,
                                            null_idx=0),
        "cls_emb": lit_types.Embeddings()
    }
    # Gradients, if requested.
    if self.compute_grads:
      ret["token_grad_sentence"] = lit_types.TokenGradients(
          align="tokens")

    # Attention heads, one field for each layer.
    for i in range(self.model.config.num_hidden_layers):
      ret[f"layer_{i}/attention"] = lit_types.AttentionHeads(
          align_in="tokens", align_out="tokens")
    return ret


def get_wsgi_app():
  """Returns a LitApp instance for consumption by gunicorn."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(_):
  # Normally path is a directory; if it's an archive file, download and
  # extract to the transformers cache.
  model_path = FLAGS.model_path
  if model_path.endswith(".tar.gz"):
    model_path = transformers.file_utils.cached_path(
        model_path, extract_compressed_file=True)

  # Load the model we defined above.
  models = {"sst": SimpleSentimentModel(model_path)}
  # Load SST-2 validation set from TFDS.
  datasets = {"sst_dev": glue.SST2Data("validation")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
