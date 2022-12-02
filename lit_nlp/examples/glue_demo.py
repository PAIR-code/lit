r"""Example demo loading a handful of GLUE models.

For a quick-start set of models, run:
  python -m lit_nlp.examples.glue_demo \
    --quickstart --port=5432

To run with the 'normal' defaults, including full-size BERT models:
  python -m lit_nlp.examples.glue_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
import transformers  # for path caching

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_QUICKSTART = flags.DEFINE_bool(
    "quickstart", False,
    "Quick-start mode, loads smaller models and a subset of the full data.")

_MODELS = flags.DEFINE_list(
    "models", [
        "sst2-tiny:sst2:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz",
        "sst2-base:sst2:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_base.tar.gz",
        "stsb:stsb:https://storage.googleapis.com/what-if-tool-resources/lit-models/stsb_base.tar.gz",
        "mnli:mnli:https://storage.googleapis.com/what-if-tool-resources/lit-models/mnli_base.tar.gz",
    ], "List of models to load, as <name>:<task>:<path>. "
    "See MODELS_BY_TASK for available tasks. Path should be the output of "
    "saving a transformers model, e.g. model.save_pretrained(path) and "
    "tokenizer.save_pretrained(path). Remote .tar.gz files will be downloaded "
    "and cached locally.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. "
    "Note: MNLI eval set is 10k examples, so will take a while to run and may "
    "be slow on older machines. Set --max_examples=200 for a quick start.")

MODELS_BY_TASK = {
    "sst2": glue_models.SST2Model,
    "stsb": glue_models.STSBModel,
    "mnli": glue_models.MNLIModel,
}

# Pre-specified set of small models, which will load and run much faster.
QUICK_START_MODELS = (
    "sst2-tiny:sst2:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz",
    "sst2-small:sst2:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_small.tar.gz",
    "stsb-tiny:stsb:https://storage.googleapis.com/what-if-tool-resources/lit-models/stsb_tiny.tar.gz",
    "mnli-small:mnli:https://storage.googleapis.com/what-if-tool-resources/lit-models/mnli_small.tar.gz",
)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("glue_demo:get_wsgi_app() called with unused args: %s", unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Quick-start mode.
  if _QUICKSTART.value:
    FLAGS.models = QUICK_START_MODELS  # smaller, faster models
    if _MAX_EXAMPLES.value is None or _MAX_EXAMPLES.value > 1000:
      FLAGS.max_examples = 1000  # truncate larger eval sets
    logging.info("Quick-start mode; overriding --models and --max_examples.")

  models = {}
  datasets = {}

  tasks_to_load = set()
  for model_string in _MODELS.value:
    # Only split on the first two ':', because path may be a URL
    # containing 'https://'
    name, task, path = model_string.split(":", 2)
    logging.info("Loading model '%s' for task '%s' from '%s'", name, task, path)
    # Normally path is a directory; if it's an archive file, download and
    # extract to the transformers cache.
    if path.endswith(".tar.gz"):
      path = transformers.file_utils.cached_path(
          path, extract_compressed_file=True)
    # Load the model from disk.
    models[name] = MODELS_BY_TASK[task](path)
    tasks_to_load.add(task)

  ##
  # Load datasets for each task that we have a model for
  if "sst2" in tasks_to_load:
    logging.info("Loading data for SST-2 task.")
    datasets["sst_dev"] = glue.SST2Data("validation")

  if "stsb" in tasks_to_load:
    logging.info("Loading data for STS-B task.")
    datasets["stsb_dev"] = glue.STSBData("validation")

  if "mnli" in tasks_to_load:
    logging.info("Loading data for MultiNLI task.")
    datasets["mnli_dev"] = glue.MNLIData("validation_matched")
    datasets["mnli_dev_mm"] = glue.MNLIData("validation_mismatched")

  # Truncate datasets if --max_examples is set.
  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("  truncated to %d examples", len(datasets[name]))

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
