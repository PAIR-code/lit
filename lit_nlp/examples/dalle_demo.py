r"""Example dalle demo loading a custom model.

To run locally with a small number of examples:
  python -m lit_nlp.examples.dalle_demo \
      --alsologtostderr --port=5432

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
from lit_nlp.api import layout
from lit_nlp.examples.datasets import dalle_prompt
from lit_nlp.examples.models import dalle

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODELS = flags.DEFINE_list(
    "models", ["dalle-mini/dalle-mini/mega-1-fp16:latest", 'dalle-mini/dalle-mini/mini-1:v0'],
    "Models to load")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 10,
    "Maximum number of examples to load from each evaluation set. Set to None to load the full set."
)

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
Dalle_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
            modules.SliceModule,
            modules.ColorModule,
        ]
    },
    lower={
        "Predictions": [
            # only generated images for now
            modules. GeneratedImageModule,
        ],
    },
    description="Custom layout for Text to Image models.",
)
CUSTOM_LAYOUTS = {"Dalle_LAYOUT": Dalle_LAYOUT}

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
    model_name = os.path.basename(model_name_or_path)
    models[model_name] = dalle.DalleModel(model_name=model_name_or_path, predictions=1)
 
  datasets = {
      "Dalle_prompt": dalle_prompt.Dalle(),

  }

  for name in datasets:
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))


  lit_demo = dev_server.Server(
      models,
      datasets,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
