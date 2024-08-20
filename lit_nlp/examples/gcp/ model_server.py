r"""A model server for serving models on GCP via Gunicorn."""

from collections.abc import Sequence
import functools
import os
from typing import Optional
from absl import app
from absl import flags
from lit_nlp import dev_server
from lit_nlp.examples.prompt_debugging import models as prompt_debugging_models
from lit_nlp.lib import serialize
from lit_nlp.lib import wsgi_app

_FLAGS = flags.FLAGS

DEFAULT_DL_FRAMEWORK = 'kerasnlp'
DEFAULT_DL_RUNTIME = 'tensorflow'
DEFAULT_PRECISION = 'bfloat16'
DEFAULT_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODELS = 'gemma_1.1_instruct_2b_en:/cns/je-d/home/mattdangerw/keras/gemma/gemma_1.1_instruct_2b_en/3/'


def get_wsgi_app() -> wsgi_app.App:
  """Return WSGI app for container-hosted demos."""

  def wrap_handler(predict_fn):
    @functools.wraps(predict_fn)
    def _handler(app, request, environ):
      data = serialize.from_json(request.data) if len(request.data) else None
      inputs = data['inputs']
      outputs = predict_fn(inputs)
      response_body = serialize.to_json(outputs, simple=True)
      return app.respond(request, response_body, 'application/json', 200)

    return _handler

  model_config = os.getenv('MODEL_CONFIG', DEFAULT_MODELS).split(',')
  dl_framework = os.environ.get('DL_FRAMEWORK', DEFAULT_DL_FRAMEWORK)
  dl_runtime = os.environ.get('DL_RUNTIME', DEFAULT_DL_RUNTIME)
  precision = os.environ.get('PRECISION', DEFAULT_PRECISION)
  batch_size = os.environ.get('BATCH_SIZE', DEFAULT_BATCH_SIZE)
  sequence_length = os.environ.get('SEQUENCE_LENGTH', DEFAULT_SEQUENCE_LENGTH)

  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  models = prompt_debugging_models.get_models(
      models_config=model_config,
      dl_framework=dl_framework,
      dl_runtime=dl_runtime,
      precision=precision,
      batch_size=batch_size,
      sequence_length=sequence_length,
  )

  if len(DEFAULT_MODELS) < 1:
    raise ValueError('No models specified in DEFAULT_MODELS')
  model_name = DEFAULT_MODELS[0].split(':')[0]

  predict_model = models[model_name]
  salience_model = models[f'{model_name}_salience']
  tokenize_model = models[f'{model_name}_tokenize']

  handlers = {
      '/predict': predict_model.predict,
      '/salience': salience_model.predict,
      '/tokenize': tokenize_model.predict,
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
