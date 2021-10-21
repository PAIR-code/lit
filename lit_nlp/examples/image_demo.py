"""LIT demo for image model.

To run:
  python -m lit_nlp.examples.image_demo --port=5432

Then navigate to localhost:5432 to access the demo UI.
"""

from absl import app
from absl import flags
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.components import image_gradient_maps

from lit_nlp.examples.datasets import open_images
from lit_nlp.examples.models import mobilenet


FLAGS = flags.FLAGS

FLAGS.set_default('development_demo', True)
FLAGS.set_default('warm_start', 1)
FLAGS.set_default('default_layout', 'demo_layout')
FLAGS.set_default('page_title', 'LIT Image Demo')


def main(_):
  demo_layout = lit_dtypes.LitComponentLayout(
      components={
          'Main': [
              'data-table-module',
              'datapoint-editor-module',
              'lit-slice-module',
              'color-module',
          ],
          'Predictions': ['classification-module', 'scalar-module'],
          'Explanations': [
              'classification-module', 'salience-map-module'],
      },
      description='Basic layout for image demo',
  )
  datasets = {'open_images': open_images.OpenImagesDataset()}
  models = {'mobilenet': mobilenet.MobileNet()}
  interpreters = {
      'Grad': image_gradient_maps.VanillaGradients(),
      'Integrated Gradients': image_gradient_maps.IntegratedGradients(),
      'Blur IG': image_gradient_maps.BlurIG(),
      'Guided IG': image_gradient_maps.GuidedIG(),
      'XRAI': image_gradient_maps.XRAI(),
      'XRAI GIG': image_gradient_maps.XRAIGIG(),
  }

  lit_demo = dev_server.Server(models, datasets, interpreters=interpreters,
                               generators={},
                               layouts={'demo_layout': demo_layout},
                               **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == '__main__':
  app.run(main)
