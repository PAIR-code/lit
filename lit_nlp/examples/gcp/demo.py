r"""A blank demo ready to load generative models and datasets.

To use with VertexAI Model Garden models, you must install the following packages:

  pip install vertexai>=1.49.0

To run the demo, you must set you GCP project location and project id. Also, the
credential must be set using the VertexAI API key.
(https://ai.google.dev/gemini-api/docs/quickstart?lang=python#set_up_your_api_key).

You can also configure the datasets and max_examples to load. If datasets and
max_examples are not provided, the default datasets and max_examples will be used.

This LIT demo does not preload any model by default, and you can either load
them in the UI (using "Configure" menu) or preload them at demo launch time with
the optional "--gemini_model" or "--generative_model_endpoints" flags.

The "--gemini_model" flag allows loading gemini models only and is a list of
strings with format as <any_name_provided_by_you>:<accepted_gemini_version>.
The accepted Gemini version can be found at
https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference.

The "--generative_model_endpoints" flag allows loading self-hosted generative
models, and is a list of strings with format as
<any_name_provided_by_you>:<endpoint_name>. The endpoint name is the
fully-qualified endpoint resource name or endpoint ID of aiplatform. The
endpoint will be available after the model is deployed on VertexAI in GCP. The
details of endpoint can be found at
https://cloud.google.com/vertex-ai/docs/pipelines/model-endpoint-component.

The following command can be used to run the demo:
  python -m lit_nlp.examples.gcp.demo \
    --project_id=$GCP_PROJECT_ID \
    --project_location=$GCP_PROJECT_LOCATION \
    --gemini_models=$GEMINI_MODEL_NAMES \
    --generative_model_endpoints=$GENERATIVE_MODEL_ENDPOINTS \
    --datasets=$DATASETS \
    --max_examples=$MAX_EXAMPLES \
    --alsologtostderr

Then navigate to localhost:5432 to access the demo UI.
"""

from collections.abc import Sequence
import sys
from typing import Optional
from absl import app
from absl import flags
from absl import logging
from google.cloud.aiplatform import vertexai
from lit_nlp import app as lit_app
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.gcp import models as gcp_models
from lit_nlp.examples.prompt_debugging import datasets as prompt_debugging_datasets

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
_GEMINI_MODELS = flags.DEFINE_list(
    'gemini_models',
    None,
    ' List of gemini models to load, as <name>:<accepted_gemini_version>.'
    ' The accepted Gemini version can be found at'
    ' https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference.',
    required=False,
)
_GENERATIVE_MODEL_ENDPOINTS = flags.DEFINE_list(
    'generative_model_endpoints',
    None,
    (
        ' List of fully-qualified endpoint resource name or endpoint ID of'
        ' aiplatform, as <name>:<endpoint_name>.'
        ' The endpoint will be provided after the model is deployed'
        ' on VertexAI in GCP. The details of endpoint can be found at'
        ' https://cloud.google.com/vertex-ai/docs/pipelines/model-endpoint-component.'
    ),
    required=False,
)

# Define dataset information.
_DATASETS = flags.DEFINE_list(
    'datasets',
    prompt_debugging_datasets.DEFAULT_DATASETS,
    'Datasets to load, as <name>:<path>. Format should be either .jsonl where'
    " each record contains 'prompt' and optional 'target' and 'source' fields,"
    ' or a plain text file with one prompt per line.',
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    prompt_debugging_datasets.DEFAULT_MAX_EXAMPLES,
    (
        'Maximum number of examples to load from each evaluation set. Set to'
        ' None to load the full set.'
    ),
)


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

  vertexai.init(project=PROJECT_ID.value, location=LOCATION.value)

  models = {}
  if _GEMINI_MODELS.value:
    for model_string in _GEMINI_MODELS.value:
      name, gemini_model = model_string.split(':', 1)
      models[name] = gcp_models.VertexModelGardenModel(gemini_model)
  if _GENERATIVE_MODEL_ENDPOINTS.value:
    for endpoint_string in _GENERATIVE_MODEL_ENDPOINTS.value:
      name, endpoint_name = endpoint_string.split(':', 1)
      models[name] = gcp_models.SelfHostedGenerativeModel(
          aip_endpoint_name=endpoint_name,
      )

  model_loaders: lit_app.ModelLoadersMap = {}
  model_loaders['gemini'] = (
      gcp_models.VertexModelGardenModel,
      gcp_models.VertexModelGardenModel.init_spec(),
  )
  model_loaders['self_hosted_generative_model'] = (
      gcp_models.SelfHostedGenerativeModel,
      gcp_models.SelfHostedGenerativeModel.init_spec(),
  )

  datasets = prompt_debugging_datasets.get_datasets(
      datasets_config=_DATASETS.value, max_examples=_MAX_EXAMPLES.value
  )
  dataset_loaders = prompt_debugging_datasets.get_dataset_loaders()

  # TODO(faneycourage): Design and add a layout for generative demos.
  lit_demo = dev_server.Server(
      models=models,
      model_loaders=model_loaders,
      datasets=datasets,
      dataset_loaders=dataset_loaders,
      **server_flags.get_flags()
  )
  return lit_demo.serve()


if __name__ == '__main__':
  app.run(main)
