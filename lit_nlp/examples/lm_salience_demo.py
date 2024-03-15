r"""Demo for sequence salience with a left-to-right language model.

To use with Gemma models, install the latest versions of Keras and KerasNLP:

  pip install keras>=3.0.5 keras-nlp>=0.8.0

To run:
  blaze run -c opt examples:lm_salience_demo -- \
    --models=gemma_instruct_2b_en:gemma_instruct_2b_en \
    --port=8890 --alsologtostderr

We strongly recommend a GPU or other accelerator to run this demo, although for
testing the smaller GPT-2 models run well on CPU; use
--models=gpt2:https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz

By default this include a small set of sample prompts, but you can load your
own examples using the --datasets flag or through the "Configure" menu in the
UI.
"""

from collections.abc import Sequence
import functools
import os
import re
import sys
from typing import Optional

# TODO(b/327281789): remove once keras 3 is the default.
# Temporary; need to set this before importing keras_nlp
os.environ["FORCE_KERAS_3"] = "True"

# pylint: disable=g-import-not-at-top
from absl import app
from absl import flags
from absl import logging
import keras
from keras_nlp import models as keras_models
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.datasets import lm as lm_data
from lit_nlp.examples.models import instrumented_keras_lms as lit_keras
from lit_nlp.examples.models import pretrained_lms
from lit_nlp.lib import file_cache

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models",
    [
        "gemma_instruct_2b_en:gemma_instruct_2b_en",
        "gpt2:https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz",
    ],
    "Models to load, as <name>:<path>. Currently supports Gemma and GPT-2"
    " variants.",
)

_DATASETS = flags.DEFINE_list(
    "datasets",
    ["sample_prompts"],
    "Datasets to load, as <name>:<path>. Format should be either .jsonl where"
    " each record contains 'prompt' and optional 'target' and optional"
    " 'source', or .txt with one prompt per line.",
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

# TODO(lit-dev): move these layouts to a separate .py file.
# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
LEFT_RIGHT_LAYOUT = layout.LitCanonicalLayout(
    left={
        "Examples": [modules.DataTableModule],
        "Editor": [modules.SingleDatapointEditorModule],
    },
    upper={  # if 'lower' not specified, this fills the right side
        "Salience": [modules.LMSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(leftWidth=40),
    description="Left/right layout for language model salience.",
)
TOP_BOTTOM_LAYOUT = layout.LitCanonicalLayout(
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
THREE_PANEL_LAYOUT = layout.LitCanonicalLayout(
    left={
        "Data Table": [modules.DataTableModule],
        "Embeddings": [modules.EmbeddingsModule],
    },
    upper={
        "Datapoint Editor": [modules.SingleDatapointEditorModule],
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

CUSTOM_LAYOUTS = {
    "left_right": LEFT_RIGHT_LAYOUT,
    "top_bottom": TOP_BOTTOM_LAYOUT,
    "three_panel": THREE_PANEL_LAYOUT,
}

FLAGS.set_default("page_title", "LM Salience Demo")
FLAGS.set_default("default_layout", "left_right")

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
  if hasattr(keras, "config") and hasattr(keras.config, "set_floatx"):
    keras.config.set_floatx(_KERAS_FLOATX.value)
  else:
    # TODO(b/327281789): remove once we can guarantee Keras 3.
    logging.warn(
        "keras.config.set_floatx() not available; using default precision."
    )

  plaintextPrompts = functools.partial(  # pylint: disable=invalid-name
      lm_data.PlaintextSents, field_name="prompt"
  )
  # Hack: normally dataset loaders are a class object which has a __name__,
  # rather than a functools.partial
  plaintextPrompts.__name__ = "PlaintextSents"

  # Pre-loaded datasets.
  datasets = {}
  for dataset_string in _DATASETS.value:
    if dataset_string == "sample_prompts":
      dataset_name = "sample_prompts"
      path = lm_data.PromptExamples.SAMPLE_DATA_PATH
    else:
      # Only split on the first ':', because path may be a URL
      # containing 'https://'
      dataset_name, path = dataset_string.split(":", 1)
    logging.info("Loading dataset '%s' from '%s'", dataset_name, path)

    if path.endswith(".jsonl"):
      datasets[dataset_name] = lm_data.PromptExamples(path)
    # .txt or .txt-#####-of-#####
    elif path.endswith(".txt") or re.match(r".*\.txt-\d{5}-of-\d{5}$", path):
      datasets[dataset_name] = plaintextPrompts(path)
    else:
      raise ValueError(f"Unsupported dataset format for {dataset_string}")

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
      models[model_name] = pretrained_lms.HFGenerativeModel(path)
      # Salience wrapper, using same underlying Keras models so as not to
      # load the weights twice.
      models[f"_{model_name}_salience"] = (
          pretrained_lms.HFSalienceModel.from_loaded(models[model_name])
      )
      models[f"_{model_name}_tokenizer"] = (
          pretrained_lms.HFTokenizerModel.from_loaded(models[model_name])
      )
    elif model_name.startswith("gemma"):
      path = file_cache.cached_path(
          path,
          extract_compressed_file=path.endswith(".tar.gz"),
          copy_directories=True,
      )
      # Load the weights once for the underlying Keras model.
      gemma_keras_model = keras_models.GemmaCausalLM.from_preset(path)  # pytype: disable=module-attr
      models = models | lit_keras.initialize_model_group_for_salience(
          model_name, gemma_keras_model, max_length=512, batch_size=4
      )
      # Disable embeddings from the generation model.
      # TODO(lit-dev): re-enable embeddings if we can figure out why UMAP was
      # crashing? Maybe need n > 2 examples.
      models[model_name].output_embeddings = False
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
