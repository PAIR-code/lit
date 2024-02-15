"""Demo for sequence salience with a left-to-right language model."""

from collections.abc import Sequence
import functools
import os
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import keras
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.datasets import lm as lm_data
from lit_nlp.examples.models import pretrained_lms

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models",
    [
        "gpt2:https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz",
        "distilgpt2:https://storage.googleapis.com/what-if-tool-resources/lit-models/distilgpt2.tar.gz",
    ],
    "Models to load, as <name>:<path>. Currently supports GPT-2 variants.",
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    1000,
    (
        "Maximum number of examples to load from each evaluation set. Set to"
        " None to load the full set."
    ),
)

_KERAS_FLOATX = flags.DEFINE_string(
    "keras_floatx", "bfloat16", "Floating-point type for Keras models."
)

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
LM_LAYOUT = layout.LitCanonicalLayout(
    left={
        "Data Table": [modules.DataTableModule],
        "Embeddings": [modules.EmbeddingsModule],
    },
    upper={
        "Datapoint Editor": [modules.DatapointEditorModule],
        "Datapoint Generators": [modules.GeneratorModule],
    },
    lower={
        "Salience": [modules.LMSalienceModule],
        "Metrics": [modules.MetricsModule],
    },
    layoutSettings=layout.LayoutSettings(
        mainHeight=40,
        leftWidth=40,
    ),
    description="Custom layout for language model salience.",
)
SIMPLE_LM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Examples": [modules.SimpleDataTableModule],
        "Editor": [modules.SimpleDatapointEditorModule],
    },
    lower={
        "Salience": [modules.LMSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(
        hideToolbar=True,
        mainHeight=40,
        centerPage=True,
    ),
    description="Simplified layout for language model salience.",
)

CUSTOM_LAYOUTS = {
    "simple": SIMPLE_LM_LAYOUT,
    "three_panel": LM_LAYOUT,
}

FLAGS.set_default("page_title", "LM Salience Demo")
FLAGS.set_default("default_layout", "simple")

_SPLASH_SCREEN_DOC = """
# Language Model Salience

To begin, select an example, then click the segment(s) (tokens, words, etc.) 
of the output that you would like to explain. Preceding segments(s) will be 
highlighted according to their importance to the selected target segment(s), 
with darker colors indicating a greater influence (salience) of that segment on
the model's likelihood of the target segment.
"""


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

  # Set Keras backend and floating-point precision.
  os.environ["KERAS_BACKEND"] = "tensorflow"
  keras.config.set_floatx(_KERAS_FLOATX.value)

  plaintextPrompts = functools.partial(  # pylint: disable=invalid-name
      lm_data.PlaintextSents, field_name="prompt"
  )
  # Hack: normally dataset loaders are a class object which has a __name__,
  # rather than a functools.partial
  plaintextPrompts.__name__ = "PlaintextSents"

  # Pre-loaded datasets.
  datasets = {
      "sample_prompts": lm_data.PromptExamples(
          lm_data.PromptExamples.SAMPLE_DATA_PATH
      ),
  }

  # For loading from the UI.
  dataset_loaders = {
      "jsonl_examples": (
          lm_data.PromptExamples,
          lm_data.PromptExamples.init_spec(),
      ),
      "plaintext_inputs": (
          plaintextPrompts,
          lm_data.PlaintextSents.init_spec(),
      ),
  }

  ##
  # Load models, according to the --models flag.
  models = {}
  for model_string in _MODELS.value:
    # Only split on the first ':', because path may be a URL
    # containing 'https://'
    model_name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", model_name, path)
    if model_name.startswith("gpt2") or model_name in ["distilgpt2"]:
      models[model_name] = pretrained_lms.GPT2GenerativeModel(path)
      # Salience wrapper, using same underlying Keras models so as not to
      # load the weights twice.
      models[f"_{model_name}_salience"] = (
          pretrained_lms.GPT2SalienceModel.from_loaded(models[model_name])
      )
      models[f"_{model_name}_tokenizer"] = (
          pretrained_lms.GPT2TokenizerModel.from_loaded(models[model_name])
      )
    else:
      raise ValueError(
          f"Unsupported model name '{model_name}' from path '{path}'"
      )

  for name in datasets:
    datasets[name] = datasets[name].slice[: _MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  lit_demo = dev_server.Server(
      models,
      datasets,
      layouts=CUSTOM_LAYOUTS,
      dataset_loaders=dataset_loaders,
      onboard_start_doc=_SPLASH_SCREEN_DOC,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
