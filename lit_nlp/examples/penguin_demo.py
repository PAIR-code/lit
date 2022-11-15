"""🐧 LIT demo for tabular data using penguin classification.

To run:
  python -m lit_nlp.examples.penguin_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""

import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.components import minimal_targeted_counterfactuals
from lit_nlp.examples.datasets import penguin_data
from lit_nlp.examples.models import penguin_model

MODEL_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/penguin.h5'  # pylint: disable=line-too-long
import transformers
MODEL_PATH = transformers.file_utils.cached_path(MODEL_PATH)

FLAGS = flags.FLAGS
FLAGS.set_default('default_layout', 'penguins')
_MODEL_PATH = flags.DEFINE_string('model_path', MODEL_PATH,
                                  'Path to load trained model.')


# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
PENGUIN_LAYOUT = layout.LitCanonicalLayout(
    upper={
        'Main': [
            modules.DiveModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    },
    lower=layout.STANDARD_LAYOUT.lower,
    description='Custom layout for the Palmer Penguins demo.',
)
CUSTOM_LAYOUTS = {'penguins': PENGUIN_LAYOUT}


# Function for running demo through gunicorn instead of the local dev server.
def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  FLAGS.set_default('server_type', 'external')
  FLAGS.set_default('demo_mode', True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info('penguin_demo:get_wsgi_app() called with unused args: %s',
                 unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  models = {'species classifier': penguin_model.PenguinModel(_MODEL_PATH.value)}
  datasets = {'penguins': penguin_data.PenguinDataset()}
  generators = {
      'Minimal Targeted Counterfactuals':
          minimal_targeted_counterfactuals.TabularMTC()
  }
  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == '__main__':
  app.run(main)
