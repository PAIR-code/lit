r"""A blank demo ready to load generative text to image models and datasets.

To use with VertexAI Model Garden models, you must install the following packages:
  pip install vertexai>=1.49.0
To run the demo, you must set you GCP project location and project id.

Currently, the demo only supports the image generation models in the Model
Garden.

The following command can be used to run the demo:
  blaze run -c opt examples/gcp_text_to_image:demo -- \
    --project_id=$GCP_PROJECT_ID \
    --project_location=$GCP_PROJECT_LOCATION \
    --alsologtostderr
Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import google.auth
from google.cloud.aiplatform import vertexai
from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.examples.gcp_text_to_image import datasets as gcp_text_to_image_datasets
from lit_nlp.examples.gcp_text_to_image import models as gcp_text_to_image_models


FLAGS = flags.FLAGS
# Define GCP project information and vertex AI API key.
LOCATION = flags.DEFINE_string(
    'project_location',
    None,
    'Please enter your GCP project location',
    required=True,
)
PROJECT_ID = flags.DEFINE_string(
    'project_id',
    None,
    'Please enter your project id',
    required=True,
)

# Custom frontend layout; see api/layout.py
_modules = layout.LitModuleName
_IMAGE_LAYOUT = layout.LitCanonicalLayout(
    upper={
        'Main': [
            _modules.DataTableModule,
            _modules.DatapointEditorModule,
        ]
    },
    lower={
        'Predictions': [
            _modules.GeneratedImageModule,
            _modules.GeneratedTextModule,
        ],
    },
    description='Custom layout for Text to Image models.',
)


CUSTOM_LAYOUTS = layout.DEFAULT_LAYOUTS | {'_IMAGE_LAYOUT': _IMAGE_LAYOUT}

_CANNED_PROMPTS = ['I have a dream', 'I have a shiba dog named cola']


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  FLAGS.set_default('server_type', 'external')
  FLAGS.set_default('demo_mode', True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info(
        'generateive_demo:get_wsgi_app() called with unused args: %s', unused
    )
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  creds, _ = google.auth.default(
      scopes=['https://www.googleapis.com/auth/cloud-platform']
  )
  creds = creds.with_quota_project(PROJECT_ID.value)
  vertexai.init(
      project=PROJECT_ID.value,
      location=LOCATION.value,
      credentials=creds,
  )
  models = {}
  model_loaders: lit_app.ModelLoadersMap = {}
  model_loaders['text_to_image'] = (
      gcp_text_to_image_models.VertexModelGardenModel,
      gcp_text_to_image_models.VertexModelGardenModel.init_spec(),
  )

  datasets = {
      'prompts': gcp_text_to_image_datasets.TextToImageDataset(_CANNED_PROMPTS)
  }
  dataset_loaders: lit_app.DatasetLoadersMap = {}
  dataset_loaders['text_to_image'] = (
      gcp_text_to_image_datasets.TextToImageDataset,
      gcp_text_to_image_datasets.TextToImageDataset.init_spec(),
  )

  lit_demo = dev_server.Server(
      models=models,
      model_loaders=model_loaders,
      datasets=datasets,
      dataset_loaders=dataset_loaders,
      layout=layout.DEFAULT_LAYOUTS,
      **server_flags.get_flags()
  )
  return lit_demo.serve()


if __name__ == '__main__':
  app.run(main)
