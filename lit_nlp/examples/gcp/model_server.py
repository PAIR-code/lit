r"""A model server for serving models on GCP via Gunicorn."""

from collections.abc import Sequence
import functools
import os
from typing import Optional

from absl import app
from lit_nlp import dev_server
from lit_nlp.examples.gcp import constants as lit_gcp_constants
from lit_nlp.examples.prompt_debugging import models as pd_models
from lit_nlp.examples.prompt_debugging import utils as pd_utils
from lit_nlp.lib import serialize
from lit_nlp.lib import wsgi_app

DEFAULT_DL_FRAMEWORK = 'kerasnlp'
DEFAULT_DL_RUNTIME = 'tensorflow'
DEFAULT_PRECISION = 'bfloat16'
DEFAULT_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODELS = 'gemma_1.1_2b_IT:gemma_1.1_instruct_2b_en'

_LlmHTTPEndpoints = lit_gcp_constants.LlmHTTPEndpoints


def get_wsgi_app() -> wsgi_app.App:
  """Return WSGI app for an LLM server."""

  def wrap_handler(predict_fn):
    @functools.wraps(predict_fn)
    def _handler(app: wsgi_app.App, request, unused_environ):
      data = serialize.from_json(request.data) if len(request.data) else None
      inputs = data['inputs']
      outputs = predict_fn(inputs)
      response_body = serialize.to_json(list(outputs),  simple=True)
      return app.respond(request, response_body, 'application/json', 200)

    return _handler

  if not (model_config := os.getenv('MODEL_CONFIG', DEFAULT_MODELS).split(',')):
    raise ValueError('No model configuration was provided')
  elif (num_configs := len(model_config)) > 1:
    raise ValueError(
        f'Only 1 model configuration can be provided, got {num_configs}'
    )

  dl_framework = os.getenv('DL_FRAMEWORK', DEFAULT_DL_FRAMEWORK)
  dl_runtime = os.getenv('DL_RUNTIME', DEFAULT_DL_RUNTIME)
  precision = os.getenv('PRECISION', DEFAULT_PRECISION)
  batch_size = int(os.getenv('BATCH_SIZE', DEFAULT_BATCH_SIZE))
  sequence_length = int(os.getenv('SEQUENCE_LENGTH', DEFAULT_SEQUENCE_LENGTH))

  models = pd_models.get_models(
      models_config=model_config,
      dl_framework=dl_framework,
      dl_runtime=dl_runtime,
      precision=precision,
      batch_size=batch_size,
      sequence_length=sequence_length,
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

def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  return get_wsgi_app()


if __name__ == '__main__':
  app.run(main)
