r"""Example demo loading pre-trained language models.

Currently supports the following model types:
- BERT (bert-*) as a masked language model
- GPT-2 (gpt2* or distilgpt2) as a left-to-right language model

To run locally:
  python -m lit_nlp.examples.lm_demo \
      --models=bert-base-uncased --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging

from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.datasets import lm
from lit_nlp.examples.models import pretrained_lms

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models",
    [
        "bert-base-uncased:https://storage.googleapis.com/what-if-tool-resources/lit-models/bert-base-uncased.tar.gz",
        "gpt2:https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz",
    ],
    "Models to load, as <name>:<path>. Currently supports variants of BERT and"
    " GPT-2.",
)

_TOP_K = flags.DEFINE_integer(
    "top_k", 10, "Rank to which the output distribution is pruned.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    1000,
    (
        "Maximum number of examples to load from each evaluation set. Set to"
        " None to load the full set."
    ),
)

_LOAD_BWB = flags.DEFINE_bool(
    "load_bwb",
    False,
    (
        "If true, will load examples from the Billion Word Benchmark dataset."
        " This may download a lot of data the first time you run it, so disable"
        " by default for the quick-start example."
    ),
)

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
LM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            modules.LanguageModelPredictionModule,
            modules.ConfusionMatrixModule,
        ],
        "Counterfactuals": [modules.GeneratorModule],
    },
    description="Custom layout for language models.",
)

CUSTOM_LAYOUTS = dict(layout.DEFAULT_LAYOUTS, **{"lm": LM_LAYOUT})

# You can also change this via URL param e.g. localhost:5432/?layout=default
FLAGS.set_default("default_layout", "lm")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("lm_demo:get_wsgi_app() called with unused args: %s", unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ##
  # Load models, according to the --models flag.
  models = {}
  for model_string in _MODELS.value:
    # Only split on the first ':', because path may be a URL
    # containing 'https://'
    model_name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", model_name, path)
    if model_name.startswith("bert-"):
      models[model_name] = pretrained_lms.BertMLM(path, top_k=_TOP_K.value)
    elif model_name.startswith("gpt2") or model_name in ["distilgpt2"]:
      models[model_name] = pretrained_lms.GPT2LanguageModel(
          path, top_k=_TOP_K.value
      )
    else:
      raise ValueError(
          f"Unsupported model name '{model_name}' from path '{path}'"
      )

  datasets = {
      # Single sentences from movie reviews (SST dev set).
      "sst_dev": glue.SST2Data("validation").remap({"sentence": "text"}),
      # Longer passages from movie reviews (IMDB dataset, test split).
      "imdb_train": classification.IMDBData("test"),
      # Empty dataset, if you just want to type sentences into the UI.
      "blank": lm.PlaintextSents(""),
  }

  dataset_loaders: lit_app.DatasetLoadersMap = {
      "sst_dev": (glue.SST2DataForLM, glue.SST2DataForLM.init_spec()),
      "imdb_train": (
          classification.IMDBData,
          classification.IMDBData.init_spec(),
      ),
      "plain_text_sentences": (
          lm.PlaintextSents,
          lm.PlaintextSents.init_spec(),
      ),
  }

  # Guard this with a flag, because TFDS will download and process 1.67 GB
  # of data if you haven't loaded `lm1b` before.
  if _LOAD_BWB.value:
    # A few sentences from the Billion Word Benchmark (Chelba et al. 2013).
    datasets["bwb"] = lm.BillionWordBenchmark(
        "train", max_examples=_MAX_EXAMPLES.value)
    dataset_loaders["bwb"] = (
        lm.BillionWordBenchmark,
        lm.BillionWordBenchmark.init_spec(),
    )

  for name in datasets:
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  generators = {"word_replacer": word_replacer.WordReplacer()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      layouts=CUSTOM_LAYOUTS,
      dataset_loaders=dataset_loaders,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
