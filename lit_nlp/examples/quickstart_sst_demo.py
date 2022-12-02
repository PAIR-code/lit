r"""Quick-start demo for a sentiment analysis model.

This demo fine-tunes a small Transformer (BERT-tiny) on the Stanford Sentiment
Treebank (SST-2), and starts a LIT server.

To run locally:
  python -m lit_nlp.examples.quickstart_sst_demo --port=5432

Training should take less than 5 minutes on a single GPU. Once you see the
ASCII-art LIT logo, navigate to localhost:5432 to access the demo UI.
"""
import sys
import tempfile
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_ENCODER_NAME = flags.DEFINE_string(
    "encoder_name", "google/bert_uncased_L-2_H-128_A-2",
    "Encoder name to use for fine-tuning. See https://huggingface.co/models.")

_MODEL_PATH = flags.DEFINE_string("model_path", None,
                                  "Path to save trained model.")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Returns a LitApp instance for consumption by gunicorn."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info(
        "quickstart_sst_demo:get_wsgi_app() called with unused "
        "args: %s", unused)
  return main([])


def run_finetuning(train_path):
  """Fine-tune a transformer model."""
  train_data = glue.SST2Data("train")
  val_data = glue.SST2Data("validation")
  model = glue_models.SST2Model(_ENCODER_NAME.value)
  model.train(train_data.examples, validation_inputs=val_data.examples)
  model.save(train_path)


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model_path = _MODEL_PATH.value or tempfile.mkdtemp()
  logging.info("Working directory: %s", model_path)
  run_finetuning(model_path)

  # Load our trained model.
  models = {"sst": glue_models.SST2Model(model_path)}
  datasets = {"sst_dev": glue.SST2Data("validation")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
