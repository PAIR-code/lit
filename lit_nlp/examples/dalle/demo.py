r"""Example for dalle demo model.

To run locally with a small number of examples:
  python -m lit_nlp.examples.dalle_demo \
      --alsologtostderr --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""
from collections.abc import Sequence
import os
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.dalle import dataset
from lit_nlp.examples.dalle import model

# NOTE: additional flags defined in server_flags.py
_FLAGS = flags.FLAGS
_FLAGS.set_default("development_demo", True)
_FLAGS.set_default("default_layout", "DALLE_LAYOUT")

_MODELS = flags.DEFINE_list(
    "models",
    [
        "dalle-mini/dalle-mini/mega-1-fp16:latest",
        "dalle-mini/dalle-mini/mini-1:v0",
    ],
    "Models to load",
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples",
    5,
    "Maximum number of examples to load from each evaluation set. Set to None "
    "to load the full set.",
)

# Custom frontend layout; see api/layout.py
_modules = layout.LitModuleName
_DALLE_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            _modules.DataTableModule,
            _modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            _modules.GeneratedImageModule,
            _modules.GeneratedTextModule,
        ],
    },
    description="Custom layout for Text to Image models.",
)
_CUSTOM_LAYOUTS = {"DALLE_LAYOUT": _DALLE_LAYOUT}


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  _FLAGS.set_default("server_type", "external")
  _FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = _FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load models, according to the --models flag.
  models = {}
  for model_name_or_path in _MODELS.value:
    model_name = os.path.basename(model_name_or_path)
    # set number of images to generate default is 6
    models[model_name] = model.DalleModel(
        model_name=model_name_or_path, predictions=6
    )

  datasets = {"Dalle_prompt": dataset.DallePrompts()}

  for name in datasets:
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))

  lit_demo = dev_server.Server(
      models,
      datasets,
      layouts=_CUSTOM_LAYOUTS,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
