r"""Example demo loading a T5 model.

To run locally with a small number of examples:
  python -m lit_nlp.examples.tydi_demo \
      --alsologtostderr --port=5432 --max_examples=10

To run using the nearest-neighbor lookup index (warning, this will take a while
to load):
  python -m lit_nlp.examples.tydi_demo \
      --alsologtostderr --port=5432 --warm_start 1.0 \
      --use_indexer --initialize_index --data_dir=/tmp/t5_index

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import summarization
from lit_nlp.examples.models import tydi

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models", ["mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"],
    "Models to load")
_NUM_TO_GEN = flags.DEFINE_integer(
    "num_to_generate", 4, "Number of generations to produce for each input.")

_TOKEN_TOP_K = flags.DEFINE_integer(
    "token_top_k", 10, "Rank to which the output distribution is pruned.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 1000,
    "Maximum number of examples to load from each evaluation set. Set to None to load the full set."
)

_LOAD_BWB = flags.DEFINE_bool(
    "load_bwb", False,
    "If true, will load examples from the Billion Word Benchmark dataset. This may download a lot of data the first time you run it, so disable by default for the quick-start example."
)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ##
  # Load models, according to the --models flag.
  models = {}
  for model_name_or_path in _MODELS.value:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    models[model_name] = tydi.TydiModel(
          model=model_name_or_path,
          model_name=model_name_or_path,
          num_to_generate=_NUM_TO_GEN.value,
          token_top_k=_TOKEN_TOP_K.value,
          output_attention=False)

  datasets = {
      "tydi_qa": summarization.TYDIQA(
        split="validation-en", max_examples=_MAX_EXAMPLES.value),

  }

  for name in datasets:
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  generators = {"word_replacer": word_replacer.WordReplacer()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
