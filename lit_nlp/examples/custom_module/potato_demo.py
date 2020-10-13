# Lint as: python3
r"""Quick-start demo for a sentiment analysis model.

This demo fine-tunes a small Transformer (BERT-tiny) on the Stanford Sentiment
Treebank (SST-2), and starts a LIT server.

To run locally:
  python -m lit_nlp.examples.custom_module.potato_demo \
      --port=5432

Training should take less than 5 minutes on a single GPU. Once you see the
ASCII-art LIT logo, navigate to localhost:5432 to access the demo UI.
"""
import tempfile
import os
import pathlib

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "encoder_name", "google/bert_uncased_L-2_H-128_A-2",
    "Encoder name to use for fine-tuning. See https://huggingface.co/models.")

flags.DEFINE_string("model_path", None, "Path to save trained model.")

print('🔥🔥🔥🔥', os.path.join(pathlib.Path(__file__).parent.absolute(), 'build'))

# Use our custom frontend build from this directory.
FLAGS.set_default("client_root", os.path.join(pathlib.Path(__file__).parent.absolute(), 'build'))
FLAGS.set_default("default_layout", "potato")

def run_finetuning(train_path):
  """Fine-tune a transformer model."""
  train_data = glue.SST2Data("train")
  val_data = glue.SST2Data("validation")
  model = glue_models.SST2Model(FLAGS.encoder_name)
  # model.train(train_data.examples, validation_inputs=val_data.examples)
  model.save(train_path)


def main(_):
  model_path = FLAGS.model_path or tempfile.mkdtemp()
  logging.info("Working directory: %s", model_path)
  run_finetuning(model_path)

  # Load our trained model.
  models = {"sst": glue_models.SST2Model(model_path)}
  datasets = {"sst_dev": glue.SST2Data("validation")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
