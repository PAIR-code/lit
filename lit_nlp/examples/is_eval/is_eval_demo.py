r"""Example demo loading a handful of IS eval models.

For a quick-start set of models, run:
  blaze run -c opt --config=cuda examples/is_eval:is_eval_demo -- \
    --quickstart --port=5432
"""
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.is_eval import datasets
from lit_nlp.examples.models import glue_models

import transformers  # for path caching

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models", [
        "sst2_single_token:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_single_token_bert.tar.gz",
        "sst2_token_in_context:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_token_in_context_bert.tar.gz",
        "sst2_ordered_pair:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_simple_order_bert.tar.gz",
        "toxicity_single_token:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_single_token_bert.tar.gz",
        "toxicity_token_in_context:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_token_in_context_bert.tar.gz",
        "toxicity_ordered_pair:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_simple_order_bert.tar.gz",
    ], "List of models to load, as <name>:<path>. "
    "Path should be the output of saving a transformer model, e.g. "
    "model.save_pretrained(path) and tokenizer.save_pretrained(path). Remote "
    ".tar.gz files will be downloaded and cached locally.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. Set "
    "--max_examples=200 for a quick start.")

DATASETS = {
    "sst2_single_token_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_single_token-dev.100syn.tsv",
    "sst2_token_in_context_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_token_in_context-dev.100syn.tsv",
    "sst2_ordered_pair_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_simple_order-dev.100syn.tsv",
    "toxicity_single_token_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_single_token-dev.100syn.tsv",
    "toxicity_token_in_context_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_token_in_context-dev.100syn.tsv",
    "toxicity_ordered_pair_dev_100_syn":
        "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_simple_order-dev.100syn.tsv",
}


def get_wsgi_app():
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("is_eval_demo:get_wsgi_app() called with unused args: %s",
                 unused)
  return main([])


def main(_):
  models = {}
  loaded_datasets = {}

  for model_string in _MODELS.value:
    # Only split on the first two ':', because path may be a URL
    # containing 'https://'
    name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", name, path)
    # Normally path is a directory; if it's an archive file, download and
    # extract to the transformers cache.
    if path.endswith(".tar.gz"):
      path = transformers.file_utils.cached_path(
          path, extract_compressed_file=True)
    # Load the model from disk.
    models[name] = glue_models.SST2Model(path)

  logging.info("Loading data for SST-2 task.")
  for data_key, url in DATASETS.items():
    path = transformers.file_utils.cached_path(url)
    loaded_datasets[data_key] = datasets.SingleInputClassificationFromTSV(path)

  # Truncate datasets if --max_examples is set.
  for name in loaded_datasets:
    logging.info("Dataset: '%s' with %d examples", name,
                 len(loaded_datasets[name]))
    loaded_datasets[name] = loaded_datasets[name].shuffle().slice[:_MAX_EXAMPLES
                                                                  .value]
    logging.info("  truncated to %d examples", len(loaded_datasets[name]))

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, loaded_datasets,
                               **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
