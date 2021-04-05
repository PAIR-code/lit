# Lint as: python3
"""Client code for querying remote models hosted by a LIT server."""

from typing import Text, Optional, Any, List, Dict
import urllib

from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import serialize
import requests
import six

urlopen = urllib.urlopen

JsonDict = types.JsonDict


def query_lit_server(url: Text,
                     endpoint: Text,
                     params: Optional[Dict[Text, Text]] = None,
                     inputs: Optional[Any] = None,
                     config: Optional[Any] = None) -> Any:
  """Query a LIT server from Python."""
  # Pack data for LIT request
  data = {'inputs': inputs, 'config': config}
  # TODO(lit-dev): for open source, require HTTPS.
  if not url.startswith('http://'):
    url = 'http://' + url
  full_url = urllib.parse.urljoin(url, endpoint)
  # Use requests to handle URL params.
  rq = requests.Request(
      'POST',
      full_url,
      params=params,
      data=serialize.to_json(data),
      headers={'Content-Type': 'application/json'})
  rq = rq.prepare()
  # Convert to urllib request
  request = urllib.request.Request(
      url=rq.url,
      data=six.ensure_binary(rq.body) if rq.body else None,
      headers=rq.headers,
      method=rq.method)
  response = urlopen(request)
  if response.code != 200:
    raise IOError(f'Failed to query {rq.url}; response code {response.code}')
  # TODO(iftenney): handle bad server response, e.g. if corplogin is required
  # and the server sends a login page instead of a JSON response.
  response_bytes = response.read()
  return serialize.from_json(six.ensure_text(response_bytes))


class RemoteModel(lit_model.Model):
  """LIT model backed by a remote LIT server."""

  def __init__(self, url: Text, name: Text, max_minibatch_size: int = 256):
    """Initialize model wrapper from remote server.

    Args:
      url: url of LIT server
      name: name of model on the remote server
      max_minibatch_size: batch size used for remote requests
    """
    self._url = url
    self._name = name

    # Get specs
    server_info = query_lit_server(self._url, 'get_info')
    self._spec = lit_model.ModelSpec(
        **server_info['models'][self._name]['spec'])

    self._max_minibatch_size = max_minibatch_size

  def input_spec(self):
    return self._spec.input

  def output_spec(self):
    return self._spec.output

  def max_minibatch_size(self):
    return self._max_minibatch_size

  def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
    # Package data as IndexedInput with dummy ids.
    indexed_inputs = [{'id': None, 'data': d} for d in inputs]
    # Omit dataset_name to bypass remote cache.
    logging.info('Querying remote model: /get_preds on %d examples',
                 len(indexed_inputs))
    preds = query_lit_server(
        self._url,
        'get_preds',
        params={
            'model': self._name,
            'response_simple_json': False
        },
        inputs=indexed_inputs)
    logging.info('Received %d predictions from remote model.', len(preds))
    return preds


def models_from_server(url: Text, **model_kw) -> Dict[Text, RemoteModel]:
  """Wrap all the models on a given LIT server."""
  server_info = query_lit_server(url, 'get_info')
  models = {}
  for name in server_info['models']:
    models[name] = RemoteModel(url, name, **model_kw)
  return models
