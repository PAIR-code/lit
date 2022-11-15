r"""Demo for a sentiment analysis model with a custom frontend build.

This demo loads a small BERT model trained on a sentiment analysis task.
It also uses a custom frontend build, which has a fun potato module!

To run locally:
  python -m lit_nlp.examples.potato_demo --port=5432

Once you see the ASCII-art LIT logo, navigate to localhost:5432 to access the
demo UI.
"""
import os
import pathlib
import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models

import transformers

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS
FLAGS.set_default("development_demo", True)
FLAGS.set_default("default_layout", "potato")

_MODEL = flags.DEFINE_string(
    "model",
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz",
    "Path to model, as in glue_demo.py")

# Use our custom frontend build from this directory.
FLAGS.set_default(
    "client_root",
    os.path.join(pathlib.Path(__file__).parent.absolute(), "build"))

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
POTATO_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [modules.DatapointEditorModule, modules.ClassificationModule],
    },
    lower={
        "Data": [modules.DataTableModule, "potato-module"],
    },
    description="Custom layout with our spud-tastic potato module.",
)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Returns a LitApp instance for consumption by gunicorn."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("potato_demo:get_wsgi_app() called with unused args: %s",
                 unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load our trained model.
  model = _MODEL.value
  if model.endswith(".tar.gz"):
    model = transformers.file_utils.cached_path(
        model, extract_compressed_file=True)

  models = {"sst": glue_models.SST2Model(model)}
  datasets = {"sst_dev": glue.SST2Data("validation")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(
      models,
      datasets,
      layouts={"potato": POTATO_LAYOUT},
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
