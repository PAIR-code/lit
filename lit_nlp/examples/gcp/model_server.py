# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A model server for serving models on GCP via Gunicorn."""

from collections.abc import Sequence
import functools
import os

from absl import app
from lit_nlp.examples.gcp import constants as lit_gcp_constants
from lit_nlp.examples.prompt_debugging import models as pd_models
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import serialize
from lit_nlp.lib import wsgi_app

DEFAULT_MODELS = 'gemma_1.1_2b_IT:gemma_1.1_instruct_2b_en'

_LlmHTTPEndpoints = lit_gcp_constants.LlmHTTPEndpoints


def get_wsgi_app() -> wsgi_app.App:
  """Return WSGI app for an LLM server."""

  def wrap_handler(predict_fn):
    @functools.wraps(predict_fn)
    def _handler(wsgiapp: wsgi_app.App, request, unused_environ):
      data = serialize.from_json(request.data) if len(request.data) else None
      inputs = data['inputs']
      outputs = predict_fn(inputs)
      response_body = serialize.to_json(list(outputs), simple=True)
      return wsgiapp.respond(request, response_body, 'application/json', 200)

    return _handler

  if not (model_config := os.getenv('MODEL_CONFIG', DEFAULT_MODELS).split(',')):
    raise ValueError('No model configuration was provided')
  elif (num_configs := len(model_config)) > 1:
    raise ValueError(
        f'Only 1 model configuration can be provided, got {num_configs}'
    )

  dl_framework = os.getenv('DL_FRAMEWORK', pd_models.DEFAULT_DL_FRAMEWORK)
  dl_runtime = os.getenv('DL_RUNTIME', pd_models.DEFAULT_DL_RUNTIME)
  precision = os.getenv('PRECISION', pd_models.DEFAULT_PRECISION)
  batch_size = int(os.getenv('BATCH_SIZE', pd_models.DEFAULT_BATCH_SIZE))
  sequence_length = int(
      os.getenv('SEQUENCE_LENGTH', pd_models.DEFAULT_SEQUENCE_LENGTH)
  )

  models = pd_models.get_models(
      models_config=model_config,
      dl_framework=dl_framework,
      dl_runtime=dl_runtime,
      precision=precision,
      batch_size=batch_size,
      max_length=sequence_length,
  )

  gen_name = model_config[0].split(':')[0]
  sal_name, tok_name = pd_utils.generate_model_group_names(gen_name)

  handlers = {
      f'/{_LlmHTTPEndpoints.GENERATE.value}': models[gen_name].predict,
      f'/{_LlmHTTPEndpoints.SALIENCE.value}': models[sal_name].predict,
      f'/{_LlmHTTPEndpoints.TOKENIZE.value}': models[tok_name].predict,
  }

  wrapped_handlers = {
      endpoint: wrap_handler(endpoint_fn)
      for endpoint, endpoint_fn in handlers.items()
  }

  return wsgi_app.App(
      wrapped_handlers, project_root='gcp', index_file='index.html'
  )


def main(argv: Sequence[str]) -> wsgi_app.App:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  return get_wsgi_app()


if __name__ == '__main__':
  app.run(main)
