r"""Example demo loading a handful of IS eval models.

To run:
  blaze run -c opt --config=cuda examples/is_eval:is_eval_demo -- \
    --port=5432
"""
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.is_eval import datasets
from lit_nlp.examples.is_eval import models as is_eval_models
from lit_nlp.lib import file_cache

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)
FLAGS.set_default("page_title", "Input Salience Evaluation Demo")

_DOC_STRING = (
    "# Input Salience Evaluation Demo\nThis demo accompanies our "
    "[paper](https://arxiv.org/abs/2211.05485) and "
    "[blogpost](https://ai.googleblog.com/2022/12/will-you-find-these-shortcuts.html)"
    " \"Will you find these shortcuts?\". We manually inserted one out of "
    "three artificial data artifacts (shortcuts) into two datasets (SST2, "
    "Toxicity). In the \"Explanations\" tab you can observe how different "
    "input salience methods put different weights on the nonsense tokens "
    "*zeroa*, *onea*, *synt*.")

_MODELS = flags.DEFINE_list(
    "models",
    [
        "sst2_single_token:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_single_token_bert.tar.gz",
        "sst2_token_in_context:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_token_in_context_bert.tar.gz",
        "sst2_ordered_pair:https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_simple_order_bert.tar.gz",
        "toxicity_single_token:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_single_token_bert.tar.gz",
        "toxicity_token_in_context:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_token_in_context_bert.tar.gz",
        "toxicity_ordered_pair:https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity_simple_order_bert.tar.gz",
    ],
    "List of models to load, as <name>:<path>. "
    "Path should be the output of saving a transformer model, e.g. "
    "model.save_pretrained(path) and tokenizer.save_pretrained(path). Remote "
    ".tar.gz files will be downloaded and cached locally.",
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. Set "
    "--max_examples=200 for a quick start.")

DATASETS = {
    "sst2_single_token_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_single_token-dev.100syn.tsv",
    "sst2_token_in_context_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_token_in_context-dev.100syn.tsv",
    "sst2_ordered_pair_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/sst2_simple_order-dev.100syn.tsv",
    "toxicity_single_token_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_single_token-dev.100syn.tsv",
    "toxicity_token_in_context_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_token_in_context-dev.100syn.tsv",
    "toxicity_ordered_pair_dev_100_syn": "https://storage.googleapis.com/what-if-tool-resources/lit-data/toxicity_simple_order-dev.100syn.tsv",
}

modules = layout.LitModuleName
IS_EVAL_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.DocumentationModule,
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            modules.ClassificationModule,
            modules.SalienceMapModule,
            modules.ScalarModule,
        ],
        "Salience Clustering": [modules.SalienceClusteringModule],
        "Metrics": [
            modules.MetricsModule,
            modules.ConfusionMatrixModule,
            modules.CurvesModule,
            modules.ThresholderModule,
        ],
        "Counterfactuals": [
            modules.GeneratorModule,
        ],
    },
    description="Custom layout for evaluating input salience methods.")
CUSTOM_LAYOUTS = {"is_eval": IS_EVAL_LAYOUT}
# You can change this back via URL param, e.g. localhost:5432/?layout=default
FLAGS.set_default("default_layout", "is_eval")


def get_wsgi_app():
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  FLAGS.set_default("warm_start", 1.0)
  FLAGS.set_default("max_examples", 1000)
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
      path = file_cache.cached_path(
          path, extract_compressed_file=True)
    # Load the model from disk.
    models[name] = is_eval_models.ISEvalModel(
        name, path, output_attention=False)

  logging.info("Loading data for SST-2 task.")
  for data_key, url in DATASETS.items():
    path = file_cache.cached_path(url)
    loaded_datasets[data_key] = datasets.SingleInputClassificationFromTSV(
        path, data_key)

  # Truncate datasets if --max_examples is set.
  for name in loaded_datasets:
    logging.info("Dataset: '%s' with %d examples", name,
                 len(loaded_datasets[name]))
    loaded_datasets[name] = loaded_datasets[name].shuffle().slice[:_MAX_EXAMPLES
                                                                  .value]
    logging.info("  truncated to %d examples", len(loaded_datasets[name]))

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(
      models,
      loaded_datasets,
      layouts=CUSTOM_LAYOUTS,
      onboard_end_doc=_DOC_STRING,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
