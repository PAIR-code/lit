# Lint as: python3
r"""Example demo loading pre-trained language models.

Currently supports the following model types:
- BERT (bert-*) as a masked language model
- GPT-2 (gpt2* or distilgpt2) as a left-to-right language model

To run locally:
  python -m lit_nlp.examples.lm_demo \
      --models=bert-base-uncased --top_k 10 --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.datasets import lm
from lit_nlp.examples.models import pretrained_lms

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

flags.DEFINE_list(
    "models", ["bert-base-uncased", "gpt2"],
    "Models to load. Currently supports variants of BERT and GPT-2.")

flags.DEFINE_integer("top_k", 10,
                     "Rank to which the output distribution is pruned.")

flags.DEFINE_integer(
    "max_examples", 1000,
    "Maximum number of examples to load from each evaluation set. Set to None to load the full set."
)

flags.DEFINE_bool(
    "load_bwb", False,
    "If true, will load examples from the Billion Word Benchmark dataset. This may download a lot of data the first time you run it, so disable by default for the quick-start example."
)

# Custom frontend layout; see client/lib/types.ts
LM_LAYOUT = lit_dtypes.LitComponentLayout(
    components={
        "Main": [
            "embeddings-module",
            "data-table-module",
            "datapoint-editor-module",
            "lit-slice-module",
            "color-module",
        ],
        "Predictions": [
            "lm-prediction-module",
            "confusion-matrix-module",
        ],
        "Counterfactuals": ["generator-module"],
    },
    description="Custom layout for language models.",
)
CUSTOM_LAYOUTS = {"lm": LM_LAYOUT}

# You can also change this via URL param e.g. localhost:5432/?layout=default
FLAGS.set_default("default_layout", "lm")


def get_wsgi_app():
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(_):

  ##
  # Load models, according to the --models flag.
  models = {}
  for model_name_or_path in FLAGS.models:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    if model_name.startswith("bert-"):
      models[model_name] = pretrained_lms.BertMLM(
          model_name_or_path, top_k=FLAGS.top_k)
    elif model_name.startswith("gpt2") or model_name in ["distilgpt2"]:
      models[model_name] = pretrained_lms.GPT2LanguageModel(
          model_name_or_path, top_k=FLAGS.top_k)
    else:
      raise ValueError(
          f"Unsupported model name '{model_name}' from path '{model_name_or_path}'"
      )

  datasets = {
      # Single sentences from movie reviews (SST dev set).
      "sst_dev": glue.SST2Data("validation").remap({"sentence": "text"}),
      # Longer passages from movie reviews (IMDB dataset, test split).
      "imdb_train": classification.IMDBData("test"),
      # Empty dataset, if you just want to type sentences into the UI.
      "blank": lm.PlaintextSents(""),
  }
  # Guard this with a flag, because TFDS will download and process 1.67 GB
  # of data if you haven't loaded `lm1b` before.
  if FLAGS.load_bwb:
    # A few sentences from the Billion Word Benchmark (Chelba et al. 2013).
    datasets["bwb"] = lm.BillionWordBenchmark(
        "train", max_examples=FLAGS.max_examples)

  for name in datasets:
    datasets[name] = datasets[name].slice[:FLAGS.max_examples]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  generators = {"word_replacer": word_replacer.WordReplacer()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
