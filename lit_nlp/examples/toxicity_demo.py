# Lint as: python3
r"""LIT Demo for a Toxicity model.

To run locally:
  python -m lit_nlp.examples.toxicity_demo --port=5432

Once you see the ASCII-art LIT logo, navigate to localhost:5432 to access the
demo UI.
"""

import sys
from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.models import glue_models

TOXICITY_MODEL_PATH = "https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity.tar.gz"  # pylint: disable=line-too-long
import transformers
TOXICITY_MODEL_PATH = transformers.file_utils.cached_path(TOXICITY_MODEL_PATH,
extract_compressed_file=True)

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

flags.DEFINE_string("model_path", TOXICITY_MODEL_PATH,
                    "Path to save trained model.")
flags.DEFINE_integer(
    "max_examples", 1000, "Maximum number of examples to load into LIT. ")


def get_wsgi_app():
  """Returns a LitApp instance for consumption by gunicorn."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(_):
  model_path = FLAGS.model_path
  logging.info("Working directory: %s", model_path)

  # Load our trained model.
  models = {"toxicity": glue_models.ToxicityModel(model_path)}
  datasets = {"toxicity_test": classification.ToxicityData("test")}

  # Truncate datasets if --max_examples is set.
  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:FLAGS.max_examples]
    logging.info("  truncated to %d examples", len(datasets[name]))

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
