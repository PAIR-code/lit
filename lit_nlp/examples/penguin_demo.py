"""🐧 LIT demo for tabular data using penguin classification.

To run:
  python -m lit_nlp.examples.penguin_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""

import sys
from absl import app
from absl import flags
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.components import minimal_targeted_counterfactuals
from lit_nlp.examples.datasets import penguin_data
from lit_nlp.examples.models import penguin_model

MODEL_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/penguin.h5'  # pylint: disable=line-too-long
import transformers
MODEL_PATH = transformers.file_utils.cached_path(MODEL_PATH)

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', MODEL_PATH, 'Path to load trained model.')


# Function for running demo through gunicorn instead of the local dev server.
def get_wsgi_app():
  FLAGS.set_default('server_type', 'external')
  FLAGS.set_default('demo_mode', True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(_):
  model_path = FLAGS.model_path

  models = {'species classifier': penguin_model.PenguinModel(model_path)}
  datasets = {'penguins': penguin_data.PenguinDataset()}
  generators = {
      'Minimal Targeted Counterfactuals':
          minimal_targeted_counterfactuals.TabularMTC()
  }
  lit_demo = dev_server.Server(
      models, datasets, generators=generators, **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == '__main__':
  app.run(main)
