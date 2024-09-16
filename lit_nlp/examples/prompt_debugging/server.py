r"""Server for sequence salience with a left-to-right language model.

To use with the Gemma, Llama, or Mistral models, install the latest versions of
Keras, KerasNLP, and/or HuggingFace Transformers:

  pip install keras>=3.1.0 keras-nlp>=0.9.0 transformers>=4.38.0

To run with the default configuration (Gemma on TensorFlow via Keras):

  python3 -m lit_nlp.examples.prompt_debugging.server -- \
    --models=gemma:gemma_1.1_instruct_2b_en \
    --alsologtostderr

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
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.prompt_debugging import datasets
from lit_nlp.examples.prompt_debugging import layouts
from lit_nlp.examples.prompt_debugging import models


# The following flags enable command line configuration datasets.
_DATASETS = flags.DEFINE_list(
    "datasets",
    datasets.DEFAULT_DATASETS,
    "Datasets to load, as <name>:<path>. Format should be either .jsonl where"
    " each record contains 'prompt' and optional 'target' and 'source' fields,"
    " or a plain text file with one prompt per line.",
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    datasets.DEFAULT_MAX_EXAMPLES,
    (
        "Maximum number of examples to load from each evaluation set. Set to"
        " None to load the full set."
    ),
)

# The following flags enable command line configuration of models.
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    models.DEFAULT_BATCH_SIZE,
    "The number of examples to process per batch.",
)

_SUPPORTED_FRAMEWORKS = ("kerasnlp", "transformers")
_DL_FRAMEWORK = flags.DEFINE_enum(
    "dl_framework",
    models.DEFAULT_DL_FRAMEWORK,
    _SUPPORTED_FRAMEWORKS,
    "The deep learning framework that loads and runs the model on the backend."
    " This server will attempt to load all models specified by the --models"
    " flag with the configured framework, incompatibilities will result in"
    " errors.",
)

_DL_RUNTIME = flags.DEFINE_enum(
    "dl_runtime",
    models.DEFAULT_DL_RUNTIME,
    # TODO(b/333373960): Add "jax" once JAX salience is supported.
    ("tensorflow", "torch"),
    "The deep learning backend framework that the model runs on. All models"
    " loaded by this server will use the same backend, incompatibilities will"
    " result in errors.",
)

_MODELS = flags.DEFINE_list(
    "models",
    models.DEFAULT_MODELS,
    "Models to load, as <name>:<path>. Path can be a URL, a local file path, or"
    " the name of a preset for the configured Deep Learning framework (either"
    " KerasNLP or HuggingFace Transformers; see --dl_framework for more). This"
    " demo is tested with Gemma, GPT2, Llama, and Mistral on all supported"
    " --dl_framework values. Other models should work, but adjustments might be"
    " needed on their tokenizers (e.g., to define custom pad_token"
    " when eos_token is not available to use as pad_token).",
)

_PRECISION = flags.DEFINE_enum(
    "precision",
    models.DEFAULT_PRECISION,
    ("bfloat16", "float32"),
    "Floating point precision for the models, only `bfloat16` and `float32` are"
    " supported at this time.",
)

_SEQUENCE_LENGTH = flags.DEFINE_integer(
    "sequence_length",
    models.DEFAULT_SEQUENCE_LENGTH,
    "The maximum sequence length of the input prompt + generated text",
)

_FLAGS = flags.FLAGS
_FLAGS.set_default("development_demo", True)
_FLAGS.set_default("page_title", "LM Prompt Debugging")
_FLAGS.set_default("default_layout", layouts.THREE_PANEL)

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
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("lm_demo:get_wsgi_app() called with unused args: %s", unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  lit_demo = dev_server.Server(
      models=models.get_models(
          models_config=_MODELS.value,
          dl_framework=_DL_FRAMEWORK.value,
          dl_runtime=_DL_RUNTIME.value,
          precision=_PRECISION.value,
          batch_size=_BATCH_SIZE.value,
          max_length=_SEQUENCE_LENGTH.value,
      ),
      datasets=datasets.get_datasets(
          datasets_config=_DATASETS.value, max_examples=_MAX_EXAMPLES.value
      ),
      layouts=layouts.PROMPT_DEBUGGING_LAYOUTS,
      model_loaders=models.get_model_loaders(
          dl_framework=_DL_FRAMEWORK.value,
          dl_runtime=_DL_RUNTIME.value,
          batch_size=_BATCH_SIZE.value,
          max_length=_SEQUENCE_LENGTH.value,
      ),
      dataset_loaders=datasets.get_dataset_loaders(),
      onboard_start_doc=_SPLASH_SCREEN_DOC,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
