r"""Example for dalle-mini demo model.

To run locally with a small number of examples:
  python -m lit_nlp.examples.dalle_mini.demo


Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.dalle_mini import data as dalle_data
from lit_nlp.examples.dalle_mini import model as dalle_model


# NOTE: additional flags defined in server_flags.py
_FLAGS = flags.FLAGS
_FLAGS.set_default("development_demo", True)
_FLAGS.set_default("default_layout", "DALLE_LAYOUT")

_FLAGS.DEFINE_integer("grid_size", 4, "The grid size to use for the model.")

_MODELS = (["dalle-mini"],)

_CANNED_PROMPTS = ["I have a dream", "I have a shiba dog named cola"]

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


CUSTOM_LAYOUTS = layout.DEFAULT_LAYOUTS | {"DALLE_LAYOUT": _DALLE_LAYOUT}


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

  model_loaders: lit_app.ModelLoadersMap = {}
  model_loaders["dalle-mini"] = (
      dalle_model.DalleMiniModel,
      dalle_model.DalleMiniModel.init_spec(),
  )

  datasets = {"examples": dalle_data.DallePrompts(_CANNED_PROMPTS)}
  dataset_loaders: lit_app.DatasetLoadersMap = {}
  dataset_loaders["text_to_image"] = (
      dalle_data.DallePrompts,
      dalle_data.DallePrompts.init_spec(),
  )

  lit_demo = dev_server.Server(
      models=models,
      model_loaders=model_loaders,
      datasets=datasets,
      dataset_loaders=dataset_loaders,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags(),
  )
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
