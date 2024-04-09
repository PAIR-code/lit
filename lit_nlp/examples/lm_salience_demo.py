r"""Demo for sequence salience with a left-to-right language model.

To use with the Gemma, Llama, or Mistral models, install the latest versions of
Keras, KerasNLP, and/or HuggingFace Transformers:

  pip install keras>=3.1.0 keras-nlp>=0.9.0 transformers>=4.38.0

To run with the default configuration (Gemma on TensorFlow via Keras):

  blaze run -c opt examples:lm_salience_demo -- \
    --models=gemma_1.1_instruct_2b_en:gemma_1.1_instruct_2b_en \
    --port=8890 --alsologtostderr

MODELS:

We strongly recommend a GPU or other accelerator to run this server with LLMs.
The table below shows the model names and presets for common models. Use these
to parameterize the --models flag with comma-separated `{model}:{preset}`
strings, and remember the number of models loaded will be limited by the memory
available on your accelerator.

| Model   | dl_framework | dl_backend=tensorflow Preset | dl_backend=torch Preset              |
| ------- | ------------ | ---------------------------- | ------------------------------------ |
| Gemma   | kerasnlp     | gemma_1.1_instruct_7b_en     | gemma_1.1_instruct_7b_en             |
| Gemma   | transformers | Unavailable                  | google/gemma-1.1-7b-it               |
| Llama 2 | kerasnlp     | llama2_instruct_7b_en        | llama2_instruct_7b_en                |
| Llama 2 | transformers | Unavailable                  | meta-llama/Llama-2-7b-hf             |
| Mistral | kerasnlp     | mistral_instruct_7b_en       | mistral_instruct_7b_en               |
| Mistral | transformers | Unavailable                  | mistralai/Mistral-7B-Instruct-v0.2   |

Additional model presets can be found at the following locations, though
compatibility with the LIT model wrappers is not guaranteed:

* KerasNLP: https://keras.io/api/keras_nlp/models/
* HuggingFace Transformers: https://huggingface.co/models

DATASETS:

By default this includes a small set of sample prompts. You can load your own
examples using the --datasets flag or through the "Configure" menu in the UI.
"""

from collections.abc import Sequence
import functools
import os
import re
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.datasets import lm as lm_data
from lit_nlp.lib import file_cache

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models",
    ["gemma_1.1_instruct_2b_en:gemma_1.1_instruct_2b_en"],
    "Models to load, as <name>:<path>. Path can be a URL, a local file path, or"
    " the name of a preset for the configured Deep Learning framework (either"
    " KerasNLP or HuggingFace Transformers; see --dl_framework for more). This"
    " demo is tested with Gemma, GPT2, Llama, and Mistral on all supported"
    " --dl_framework values. Other models should work, but adjustments might be"
    " needed on their tokenizers (e.g., to define custom pad_token"
    " when eos_token is not available to use as pad_token).",
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

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 4, "The number of examples to process per batch.",
)

_DL_BACKEND = flags.DEFINE_enum(
    "dl_backend",
    "tensorflow",
    ["jax", "torch", "tensorflow"],
    "The deep learning backend framework that the model runs on. All models"
    " loaded by this server will use the same backend, incompatibilities will"
    " result in errors.",
)

_DL_FRAMEWORK = flags.DEFINE_enum(
    "dl_framework",
    "kerasnlp",
    ["kerasnlp", "transformers"],
    "The deep learning framework that loads and runs the model on the backend."
    " This server will attempt to load all models specified by the --models"
    " flag with the configured framework, incompatibilities will result in"
    " errors.",
)

_PRECISION = flags.DEFINE_enum(
    "precision",
    "bfloat16",
    ["bfloat16", "float32"],
    "Floating point precision for the models, only `bfloat16` and `float32` are"
    " supported for now.",
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
  if _DL_FRAMEWORK.value == "kerasnlp":
    # NOTE: Keras and KerasNLP require that certain environment variables are
    # set before they are imported.
    # TODO(b/327281789): Remove FORCE_KERAS_3 once Keras 3 is the default.
    os.environ["FORCE_KERAS_3"] = "True"
    os.environ["KERAS_BACKEND"] = _DL_BACKEND.value

    # NOTE: Imported here and not at the top of the file to avoid
    # initialization issues with the environment variables above. We should also
    # import keras before any other Keras-related modules (e.g., KerasNLP or the
    # LIT wrappers) to limit the potenital for improperly configured backends.
    import keras  # pylint: disable=g-import-not-at-top

    keras.config.set_floatx(_PRECISION.value)
  elif _DL_BACKEND.value == "torch":
    # NOTE: Keras sets precision for all backends with set_floatx(), but for
    # HuggingFace Transformers with PyTorch we need to set it explicitly.
    import torch  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    torch.set_default_dtype(
        torch.bfloat16 if _PRECISION.value == "bfloat16" else torch.float32
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
    # Only split on the first ':' as path may be a URL containing 'https://'
    model_name, path = model_string.split(":", 1)
    logging.info("Loading model '%s' from '%s'", model_name, path)

    path = file_cache.cached_path(
        path,
        extract_compressed_file=path.endswith(".tar.gz"),
    )

    if _DL_FRAMEWORK.value == "kerasnlp":
      # pylint: disable=g-import-not-at-top
      from keras_nlp import models as keras_models
      from lit_nlp.examples.models import instrumented_keras_lms as lit_keras
      # pylint: enable=g-import-not-at-top
      # Load the weights once for the underlying Keras model.
      model = keras_models.CausalLM.from_preset(path)
      models |= lit_keras.initialize_model_group_for_salience(
          model_name, model, max_length=512, batch_size=_BATCH_SIZE.value
      )
      # Disable embeddings from the generation model.
      # TODO(lit-dev): re-enable embeddings if we can figure out why UMAP was
      # crashing? Maybe need n > 2 examples.
      models[model_name].output_embeddings = False
    else:
      # NOTE: (Style Deviation) Imported here to limit uncessary imports.
      from lit_nlp.examples.models import pretrained_lms  # pylint: disable=g-import-not-at-top
      # Assuming a valid decoder model name supported by
      # `transformers.AutoModelForCausalLM` is provided to "path".
      models |= pretrained_lms.initialize_model_group_for_salience(
          model_name,
          path,
          batch_size=_BATCH_SIZE.value,
          framework=_DL_BACKEND.value,
          max_new_tokens=512,
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
