r"""Example demo loading a TyDiModel.

To run locally with a small number of examples:
  python -m lit_nlp.examples.tydi_demo \
      --alsologtostderr --port=5432 --max_examples=10

Then navigate to localhost:5432 to access the demo UI.
"""
from collections.abc import Sequence
import os
import sys
from typing import Optional

from absl import app
from absl import flags

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import question_answering
from lit_nlp.examples.models import tydi

# NOTE: additional flags defined in server_flags.py
_FLAGS = flags.FLAGS

_FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models", ["mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp"],
    "Models to load")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 1000,
    "Maximum number of examples to load from each evaluation set. Set to None "
    "to load the full set."
)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", True)
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
    models[model_name] = tydi.TyDiModel(model_name=model_name_or_path)

  max_examples: int = _MAX_EXAMPLES.value
  dataset_defs: tuple[tuple[str, str]] = (
      ("TyDiQA-Multilingual", "validation"),
      ("TyDiQA-English", "validation-en"),
      ("TyDiQA-Finnish", "validation-fi"),
      ("TyDiQA-Arabic", "validation-ar"),
      ("TyDiQA-Bengali", "validation-bn"),
      ("TyDiQA-Indonesian", "validation-id"),
      ("TyDiQA-Korean", "validation-ko"),
      ("TyDiQA-Russian", "validation-ru"),
      ("TyDiQA-Swahili", "validation-sw"),
      ("TyDiQA-Telugu", "validation-te"),
  )
  datasets = {
      name: question_answering.TyDiQA(split=split, max_examples=max_examples)
      for name, split in dataset_defs
  }

  generators = {"word_replacer": word_replacer.WordReplacer()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
