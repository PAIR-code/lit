# Copyright 2020 Google LLC
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
# Lint as: python3
"""LIT backend, as a standard WSGI app."""

import collections
import functools
import glob
import os
import pickle
import random
import time
from typing import Optional, Text, List, Mapping

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import gradient_maps
from lit_nlp.components import lemon_explainer
from lit_nlp.components import lime_explainer
from lit_nlp.components import metrics
from lit_nlp.components import pca
from lit_nlp.components import projection
from lit_nlp.components import scrambler
from lit_nlp.components import umap
from lit_nlp.components import word_replacer
from lit_nlp.lib import caching
from lit_nlp.lib import serialize
from lit_nlp.lib import utils
from lit_nlp.lib import wsgi_app


JsonDict = types.JsonDict

# Export this symbol, for access from demo.py
PredsCache = caching.PredsCache


def make_handler(fn):
  """Convenience wrapper to handle args and serialization.

  This is a thin shim between server (handler, request) and model logic
  (inputs, args, outputs).

  Args:
    fn: function (JsonDict, **kw) -> JsonDict

  Returns:
    fn wrapped as a request handler
  """

  @functools.wraps(fn)
  def _handler(handler, request):
    logging.info('Request received: %s', request.full_path)
    kw = request.args.to_dict()
    # The frontend needs "simple" data (e.g. NumPy arrays converted to lists),
    # but for requests from Python we may want to use the invertible encoding
    # so that datatypes from remote models are the same as local ones.
    response_simple_json = utils.coerce_bool(
        kw.pop('response_simple_json', True))
    data = serialize.from_json(request.data) if len(request.data) else None
    outputs = fn(data, **kw)
    response_body = serialize.to_json(outputs, simple=response_simple_json)
    return handler.respond(request, response_body, 'application/json', 200)

  return _handler


class LitApp(object):
  """LIT WSGI application."""

  def _build_metadata(self):
    """Build metadata from model and dataset specs."""
    info_by_model = collections.OrderedDict()
    for name, m in self._models.items():
      mspec: lit_model.ModelSpec = m.spec()
      info = {}
      info['spec'] = {'input': mspec.input, 'output': mspec.output}
      # List compatible datasets.
      info['datasets'] = [
          dname for dname, ds in self._datasets.items()
          if mspec.is_compatible_with_dataset(ds.spec())
      ]
      if len(info['datasets']) == 0:  # pylint: disable=g-explicit-length-test
        logging.error("Error: model '%s' has no compatible datasets!", name)
      # TODO(lit-team): check generator and interpreter compatibility
      # with models, or just do this on frontend?
      info['generators'] = list(self._generators.keys())
      info['interpreters'] = list(self._interpreters.keys())
      info_by_model[name] = info

    info_by_dataset = collections.OrderedDict()
    for name, ds in self._datasets.items():
      info_by_dataset[name] = {'spec': ds.spec()}

    self._info = {
        'models': info_by_model,
        'datasets': info_by_dataset,
        # TODO(lit-team): return more spec information here?
        'generators': list(self._generators.keys()),
        'interpreters': list(self._interpreters.keys()),
        'demoMode': self._demo_mode,
        'defaultLayout': self._default_layout,
    }

  def _get_spec(self, model_name: Text):
    return self._info['models'][model_name]['spec']

  def _get_info(self, unused_data, **unused_kw):
    """Get model info and send to frontend."""
    return self._info

  def _predict(self, inputs: List[JsonDict], model_name: Text,
               dataset_name: Optional[Text]):
    """Run model predictions."""
    return self._models[model_name].predict_with_metadata(
        inputs, dataset_name=dataset_name)

  def _save_datapoints(self, data, dataset_name: Text, path: Text, **unused_kw):
    """Save datapoints to disk."""
    if self._demo_mode:
      logging.warn('Attempted to save datapoints in demo mode.')
      return None
    data = data['inputs']
    timestr = time.strftime('%Y%m%d-%H%M%S')
    file_name = dataset_name + '_' + timestr + '.pkl'
    new_file_path = os.path.join(path, file_name)
    with open(new_file_path, 'wb') as fd:
      pickle.dump(data, fd)
    return new_file_path

  def _load_datapoints(self, unused_data, dataset_name: Text, path: Text,
                       **unused_kw):
    """Load datapoints from disk."""
    if self._demo_mode:
      logging.warn('Attempted to load datapoints in demo mode.')
      return None
    search_path = os.path.join(path, dataset_name) + '*.pkl'
    datapoints = []
    files = glob.glob(search_path)
    for file_path in files:
      with open(file_path, 'rb') as fd:
        datapoints.extend(pickle.load(fd))
    return datapoints

  def _get_preds(self,
                 data,
                 model: Text,
                 dataset_name: Optional[Text] = None,
                 requested_types: Text = 'LitType',
                 **unused_kw):
    """Get model predictions.

    Args:
      data: data payload, containing 'inputs' field
      model: name of the model to run
      dataset_name: name of the active dataset
      requested_types: optional, comma-separated list of types to return

    Returns:
      List[JsonDict] containing requested fields of model predictions
    """
    preds = list(self._predict(data['inputs'], model, dataset_name))

    # Figure out what to return to the frontend.
    output_spec = self._get_spec(model)['output']
    requested_types = requested_types.split(',')
    logging.info('Requested types: %s', str(requested_types))
    ret_keys = []
    for t_name in requested_types:
      t_class = getattr(types, t_name, None)
      assert issubclass(
          t_class, types.LitType), f"Class '{t_name}' is not a valid LitType."
      ret_keys.extend(utils.find_spec_keys(output_spec, t_class))
    ret_keys = set(ret_keys)  # de-dupe

    # Return selected keys.
    logging.info('Will return keys: %s', str(ret_keys))
    # One record per input.
    ret = [utils.filter_by_keys(p, ret_keys.__contains__) for p in preds]
    return ret

  def _get_datapoint_ids(self, data):
    """Fill in unique example hashes for the provided datapoints."""
    examples = []
    for example in data['inputs']:
      example['id'] = caching.input_hash(example['data'])
      examples.append(example)
    return examples

  def _get_dataset(self, unused_data, dataset_name: Text = None):
    """Attempt to get dataset, or override with a specific path."""

    # TODO(lit-team): add functionality to load data from a given path, as
    # passed from the frontend?
    assert dataset_name is not None, 'No dataset specified.'
    # TODO(lit-team): possibly allow IDs from persisted dataset.
    return caching.add_hashes_to_input(self._datasets[dataset_name].examples)

  def _get_generated(self, data, model: Text, dataset_name: Text,
                     generator: Text, **unused_kw):
    """Generate new datapoints based on the request."""
    generator = self._generators[generator]
    #  IndexedInput[] -> Input[]
    raw_inputs = [d['data'] for d in data['inputs']]
    outs = generator.generate_all(
        raw_inputs,
        self._models[model],
        self._datasets[dataset_name],
        config=data.get('config'))
    return outs

  def _get_interpretations(self, data, model: Text, dataset_name: Text,
                           interpreter: Text, **unused_kw):
    """Run an interpretation component."""
    interpreter = self._interpreters[interpreter]
    # Pre-compute using self._predict, which looks for cached results.
    model_outputs = self._predict(data['inputs'], model, dataset_name)

    return interpreter.run_with_metadata(
        data['inputs'],
        self._models[model],
        self._datasets[dataset_name],
        model_outputs=model_outputs,
        config=data.get('config'))

  def _warm_start(self, rate: float):
    """Warm-up the predictions cache by making some model calls."""
    assert rate >= 0 and rate <= 1
    for model, model_info in self._info['models'].items():
      for dataset_name in model_info['datasets']:
        logging.info("Warm-start of model '%s' on dataset '%s'", model,
                     dataset_name)
        full_dataset = self._get_dataset([], dataset_name)
        if rate < 1:
          dataset = random.sample(full_dataset, int(len(full_dataset) * rate))
          logging.info('Partial warm-start: running on %d/%d examples.',
                       len(dataset), len(full_dataset))
        else:
          dataset = full_dataset
        _ = self._get_preds({'inputs': dataset}, model, dataset_name)

  def _warm_projections(self, interpreters: List[Text]):
    """Pre-compute UMAP/PCA projections with default arguments."""
    for model, model_info in self._info['models'].items():
      for dataset_name in model_info['datasets']:
        for field_name in utils.find_spec_keys(model_info['spec']['output'],
                                               types.Embeddings):
          config = dict(
              dataset_name=dataset_name,
              model_name=model,
              field_name=field_name,
              proj_kw={'n_components': 3})
          data = {'inputs': [], 'config': config}
          for interpreter_name in interpreters:
            _ = self._get_interpretations(
                data, model, dataset_name, interpreter=interpreter_name)

  def __init__(
      self,
      models: Mapping[Text, lit_model.Model],
      datasets: Mapping[Text, lit_dataset.Dataset],
      generators: Optional[Mapping[Text, lit_components.Generator]] = None,
      interpreters: Optional[Mapping[Text, lit_components.Interpreter]] = None,
      # General server config; see server_flags.py.
      data_dir: Optional[Text] = None,
      warm_start: float = 0.0,
      warm_projections: bool = False,
      client_root: Optional[Text] = None,
      demo_mode: bool = False,
      default_layout: str = None,
  ):
    if client_root is None:
      raise ValueError('client_root must be set on application')

    self._demo_mode = demo_mode
    self._default_layout = default_layout
    if data_dir and not os.path.isdir(data_dir):
      os.mkdir(data_dir)
    self._models = {
        name: caching.CachingModelWrapper(model, name, cache_dir=data_dir)
        for name, model in models.items()
    }
    self._datasets = datasets
    if generators is not None:
      self._generators = generators
    else:
      self._generators = {
          'scrambler': scrambler.Scrambler(),
          'word_replacer': word_replacer.WordReplacer(),
      }

    if interpreters is not None:
      self._interpreters = interpreters
    else:
      metrics_group = lit_components.ComponentGroup({
          'regression': metrics.RegressionMetrics(),
          'multiclass': metrics.MulticlassMetrics(),
          'paired': metrics.MulticlassPairedMetrics(),
          'bleu': metrics.CorpusBLEU(),
      })
      self._interpreters = {
          'grad_norm': gradient_maps.GradientNorm(),
          'lime': lime_explainer.LIME(),
          'counterfactual explainer': lemon_explainer.LEMON(),
          'metrics': metrics_group,
          # Embedding projectors expose a standard interface, but get special
          # handling so we can precompute the projections if requested.
          'pca': projection.ProjectionManager(pca.PCAModel),
          'umap': projection.ProjectionManager(umap.UmapModel),
      }

    # Information on models and datasets.
    self._build_metadata()

    # Optionally, run models to pre-populate cache.
    if warm_projections:
      logging.info(
          'Projection (dimensionality reduction) warm-start requested; '
          'will do full warm-start for all models since predictions are needed.'
      )
      warm_start = 1.0

    if warm_start > 0:
      self._warm_start(rate=warm_start)
      self.save_cache()

    # If you add a new embedding projector that should be warm-started,
    # also add it to the list here.
    # TODO(lit-dev): add some registry mechanism / automation if this grows to
    # more than 2-3 projection types.
    if warm_projections:
      self._warm_projections(['pca', 'umap'])

    handlers = {
        # Metadata endpoints.
        '/get_info': self._get_info,
        # Dataset-related endpoints.
        '/get_dataset': self._get_dataset,
        '/get_generated': self._get_generated,
        '/save_datapoints': self._save_datapoints,
        '/load_datapoints': self._load_datapoints,
        '/get_datapoint_ids': self._get_datapoint_ids,
        # Model prediction endpoints.
        '/get_preds': self._get_preds,
        '/get_interpretations': self._get_interpretations,
    }

    self._wsgi_app = wsgi_app.App(
        # Wrap endpoint fns to take (handler, request)
        handlers={k: make_handler(v) for k, v in handlers.items()},
        project_root=client_root,
        index_file='static/index.html',
    )

  def save_cache(self):
    for m in self._models.values():
      if isinstance(m, caching.CachingModelWrapper):
        m.save_cache()

  def __call__(self, environ, start_response):
    """Implementation of the WSGI interface."""
    return self._wsgi_app(environ, start_response)
