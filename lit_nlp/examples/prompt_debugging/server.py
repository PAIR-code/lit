r"""Server for sequence salience with a left-to-right language model.

To use with the Gemma, Llama, or Mistral models, install the latest versions of
HuggingFace Transformers:

  pip install transformers>=4.38.0

To run with the default configuration (Gemma on PyTorch):

  python3 -m lit_nlp.examples.prompt_debugging.server -- \
    --models=gemma:google/gemma-2b-it \
    --alsologtostderr

MODELS:

We strongly recommend a GPU or other accelerator to run this server with LLMs.
The table below shows the model names and presets for common models. Use these
to parameterize the --models flag with comma-separated `{model}:{preset}`
strings, and remember the number of models loaded will be limited by the memory
available on your accelerator.

| Model   | Preset                             |
| ------- | ---------------------------------- |
| Gemma   | google/gemma-1.1-7b-it             |
| Llama 2 | meta-llama/Llama-2-7b-hf           |
| Mistral | mistralai/Mistral-7B-Instruct-v0.2 |

Additional model presets can be found at the following location, though
compatibility with the LIT model wrappers is not guaranteed:

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

_MODELS = flags.DEFINE_list(
    "models",
    models.DEFAULT_MODELS,
    "Models to load, as <name>:<path>. Path can be a URL, a local file path, or"
    " the name of a HuggingFace Transformers model. This demo is tested with"
    " Gemma, GPT2, Llama, and Mistral. Other models should work, but"
    " adjustments might be needed on their tokenizers (e.g., to define custom"
    " pad_token when eos_token is not available to use as pad_token).",
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
          precision=_PRECISION.value,
          batch_size=_BATCH_SIZE.value,
          max_length=_SEQUENCE_LENGTH.value,
      ),
      datasets=datasets.get_datasets(
          datasets_config=_DATASETS.value, max_examples=_MAX_EXAMPLES.value
      ),
      layouts=layouts.PROMPT_DEBUGGING_LAYOUTS,
      model_loaders=models.get_model_loaders(
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
