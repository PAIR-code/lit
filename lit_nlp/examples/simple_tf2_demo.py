# Lint as: python3
r"""Code example for a custom model, using TensorFlow 2.

This demo shows how to use a custom model with LIT, in just a few lines of code.
We'll use a transformers model, with a minimal amount of code to implement the
LIT API. Compared to models/glue_models.py, this has fewer features, but the
code is more readable.

This demo is equivalent in functionality to simple_pytorch_demo.py, but uses
TensorFlow 2 instead of PyTorch. The models behave identically as far as LIT is
concerned, and the implementation is quite similar - to see changes, run:
  git diff --no-index simple_pytorch_demo.py simple_tf2_demo.py

This uses the same underlying model class
(transformers.TFAutoModelForSequenceClassification) as models/glue_models.py, so
you can load from the same weights. To train something for this demo, you can:
- Use quickstart_sst_demo.py, and set --model_path to somewhere durable
- Or: Use tools/glue_trainer.py
- Or: Use any fine-tuning code that works with transformers, such as
https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts

To run locally:
  python -m lit_nlp.examples.simple_tf2_demo \
      --port=5432 --model_path=/path/to/saved/model

Then navigate to localhost:5432 to access the demo UI.
"""
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
# Use the regular GLUE data loaders, because these are very simple already.
from lit_nlp.examples.datasets import glue
from lit_nlp.lib import utils

import tensorflow as tf
import transformers

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

flags.DEFINE_string(
    "model_path",
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz",
    "Path to trained model, in standard transformers format, e.g. as "
    "saved by model.save_pretrained() and tokenizer.save_pretrained()")


def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in TF2, with fallback to PyTorch weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from PyTorch checkpoint (from_pt=True)")
    return cls.from_pretrained(*args, from_pt=True, **kw)


class SimpleSentimentModel(lit_model.Model):
  """Simple sentiment analysis model."""

  LABELS = ["0", "1"]  # negative, positive

  def __init__(self, model_name_or_path):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        output_hidden_states=True,
        output_attentions=True,
    )
    # This is a just a regular Keras model.
    self.model = _from_pretrained(
        transformers.TFAutoModelForSequenceClassification,
        model_name_or_path,
        config=model_config)

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
        return_tensors="tf",
        add_special_tokens=True,
        max_length=128,
        padding="longest",
        truncation="longest_first")

    # Run a forward pass.
    out: transformers.modeling_tf_outputs.TFSequenceClassifierOutput = \
        self.model(encoded_input, training=False)

    # Post-process outputs.
    batched_outputs = {
        "probas": tf.nn.softmax(out.logits, axis=-1),
        "input_ids": encoded_input["input_ids"],
        "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1),
        "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
    }
    # Return as NumPy for further processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Unbatch outputs so we get one record per input example.
    for output in utils.unbatch_preds(detached_outputs):
      ntok = output.pop("ntok")
      output["tokens"] = self.tokenizer.convert_ids_to_tokens(
          output.pop("input_ids")[1:ntok - 1])
      yield output

  def input_spec(self) -> lit_types.Spec:
    return {
        "sentence": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS, required=False)
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        "tokens": lit_types.Tokens(),
        "probas": lit_types.MulticlassPreds(parent="label", vocab=self.LABELS,
                                            null_idx=0),
        "cls_emb": lit_types.Embeddings()
    }


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
