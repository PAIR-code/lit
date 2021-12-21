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

import functools
import glob
import os
import random
import time
from typing import Optional, Text, List, Mapping, Sequence, Union

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import ablation_flip
from lit_nlp.components import gradient_maps
from lit_nlp.components import hotflip
from lit_nlp.components import lemon_explainer
from lit_nlp.components import lime_explainer
from lit_nlp.components import metrics
from lit_nlp.components import model_salience
from lit_nlp.components import nearest_neighbors
from lit_nlp.components import pca
from lit_nlp.components import pdp
from lit_nlp.components import projection
from lit_nlp.components import scrambler
from lit_nlp.components import tcav
from lit_nlp.components import thresholder
from lit_nlp.components import umap
from lit_nlp.components import word_replacer
from lit_nlp.lib import caching
from lit_nlp.lib import serialize
from lit_nlp.lib import utils
from lit_nlp.lib import wsgi_app

JsonDict = types.JsonDict
Input = types.Input
IndexedInput = types.IndexedInput

# Export this symbol, for access from demo.py
PredsCache = caching.PredsCache


class LitApp(object):
  """LIT WSGI application."""

  def _build_metadata(self):
    """Build metadata from model and dataset specs."""
    model_info = {}
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
      info['generators'] = [
          name for name, gen in self._generators.items() if gen.is_compatible(m)
      ]
      info['interpreters'] = [
          name for name, interp in self._interpreters.items()
          if interp.is_compatible(m)
      ]
      info['description'] = m.description()
      model_info[name] = info

    dataset_info = {}
    for name, ds in self._datasets.items():
      dataset_info[name] = {
          'spec': ds.spec(),
          'description': ds.description(),
      }

    generator_info = {}
    for name, gen in self._generators.items():
      generator_info[name] = {
          'configSpec': gen.config_spec(),
          'metaSpec': gen.meta_spec(),
          'description': gen.description()
      }

    interpreter_info = {}
    for name, interpreter in self._interpreters.items():
      interpreter_info[name] = {
          'configSpec': interpreter.config_spec(),
          'metaSpec': interpreter.meta_spec(),
          'description': interpreter.description()
      }

    return {
        # Component info and specs
        'models': model_info,
        'datasets': dataset_info,
        'generators': generator_info,
        'interpreters': interpreter_info,
        'layouts': self._layouts,
        # Global configuration
        'demoMode': self._demo_mode,
        'defaultLayout': self._default_layout,
        'canonicalURL': self._canonical_url,
        'pageTitle': self._page_title,
    }

  def _get_model_spec(self, name: Text):
    return self._info['models'][name]['spec']

  def _get_info(self, unused_data, **unused_kw):
    """Get model info and send to frontend."""
    return self._info

  def _reconstitute_inputs(self, inputs: Sequence[Union[IndexedInput, str]],
                           dataset_name: str) -> List[IndexedInput]:
    """Reconstitute any inputs sent as references (bare IDs)."""
    index = self._datasets[dataset_name].index
    # TODO(b/178228238): set up proper debug logging and hide this by default.
    num_aliased = sum([isinstance(ex, str) for ex in inputs])
    logging.info(
        "%d of %d inputs sent as IDs; reconstituting from dataset '%s'",
        num_aliased, len(inputs), dataset_name)
    return [index[ex] if isinstance(ex, str) else ex for ex in inputs]

  def _predict(self, inputs: List[JsonDict], model_name: Text,
               dataset_name: Optional[Text]):
    """Run model predictions."""
    return list(self._models[model_name].predict_with_metadata(
        inputs, dataset_name=dataset_name))

  def _save_datapoints(self, data, dataset_name: Text, path: Text, **unused_kw):
    """Save datapoints to disk."""
    if self._demo_mode:
      logging.warn('Attempted to save datapoints in demo mode.')
      return None
    return self._datasets[dataset_name].save(data['inputs'], path)

  def _load_datapoints(self, unused_data, dataset_name: Text, path: Text,
                       **unused_kw):
    """Load datapoints from disk."""
    if self._demo_mode:
      logging.warn('Attempted to load datapoints in demo mode.')
      return None
    dataset = self._datasets[dataset_name].load(path)
    return dataset.indexed_examples

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
    preds = self._predict(data['inputs'], model, dataset_name)

    # Figure out what to return to the frontend.
    output_spec = self._get_model_spec(model)['output']
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

  def _annotate_new_data(self, data, dataset_name: Optional[Text] = None,
                         **unused_kw) -> List[IndexedInput]:
    """Fill in index and other extra data for the provided datapoints."""
    # TODO(lit-dev): unify this with hash fn on dataset objects.
    assert dataset_name is not None, 'No dataset specified.'

    # Generate annotated versions of new datapoints.
    dataset = self._datasets[dataset_name]
    input_examples = [example['data'] for example in data['inputs']]
    dataset_to_annotate = lit_dataset.Dataset(
        base=dataset, examples=input_examples)
    annotated_dataset = self._run_annotators(dataset_to_annotate)

    # Add annotations and IDs to new datapoints.
    for i, example in enumerate(data['inputs']):
      example['data'] = annotated_dataset.examples[i]
      example['id'] = caching.input_hash(example['data'])

    return data['inputs']

  def _get_dataset(self,
                   unused_data,
                   dataset_name: Optional[Text] = None,
                   **unused_kw):
    """Attempt to get dataset, or override with a specific path."""
    return self._datasets[dataset_name].indexed_examples

  def _create_dataset(self,
                      unused_data,
                      dataset_name: Optional[Text] = None,
                      dataset_path: Optional[Text] = None,
                      **unused_kw):
    """Create dataset from a path, updating and returning the metadata."""

    assert dataset_name is not None, 'No dataset specified.'
    assert dataset_path is not None, 'No dataset path specified.'
    new_dataset = self._datasets[dataset_name].load(dataset_path)
    if new_dataset is not None:
      new_dataset_name = dataset_name + '-' + os.path.basename(dataset_path)
      self._datasets[new_dataset_name] = new_dataset
      self._info = self._build_metadata()
      return (self._info, new_dataset_name)
    else:
      return None

  def _create_model(self,
                    unused_data,
                    model_name: Optional[Text] = None,
                    model_path: Optional[Text] = None,
                    **unused_kw):
    """Create model from a path, updating and returning the metadata."""

    assert model_name is not None, 'No model specified.'
    assert model_path is not None, 'No model path specified.'
    # Load using the underlying model class, then wrap explicitly in a cache.
    new_model = self._models[model_name].wrapped.load(model_path)
    if new_model is not None:
      new_model_name = model_name + ':' + os.path.basename(model_path)
      self._models[new_model_name] = caching.CachingModelWrapper(
          new_model, new_model_name, cache_dir=self._data_dir)
      self._info = self._build_metadata()
      return (self._info, new_model_name)
    else:
      return None

  def _get_generated(self, data, model: Text, dataset_name: Text,
                     generator: Text, **unused_kw):
    """Generate new datapoints based on the request."""
    generator_name = generator
    generator: lit_components.Generator = self._generators[generator_name]
    dataset = self._datasets[dataset_name]
    # Nested list, containing generated examples from each input.
    all_generated: List[List[Input]] = generator.run_with_metadata(
        data['inputs'], self._models[model], dataset, config=data.get('config'))

    # Annotate datapoints
    def annotate_generated(datapoints):
      dataset_to_annotate = lit_dataset.Dataset(
          base=dataset, examples=datapoints)
      annotated_dataset = self._run_annotators(dataset_to_annotate)
      return annotated_dataset.examples

    annotated_generated = [
        annotate_generated(generated) for generated in all_generated]

    # Add metadata.
    all_generated_indexed: List[List[IndexedInput]] = [
        dataset.index_inputs(generated) for generated in annotated_generated
    ]
    for parent, indexed_generated in zip(data['inputs'], all_generated_indexed):
      for generated in indexed_generated:
        generated['meta'].update({
            'parentId': parent['id'],
            'source': generator_name,
            'added': True,
        })
    return all_generated_indexed

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
        embedding_fields = utils.find_spec_keys(model_info['spec']['output'],
                                                types.Embeddings)
        # Only warm-start on the first embedding field, since if models return
        # many different embeddings this can take a long time.
        for field_name in embedding_fields[:1]:
          config = dict(
              dataset_name=dataset_name,
              model_name=model,
              field_name=field_name,
              proj_kw={'n_components': 3})
          data = {'inputs': [], 'config': config}
          for interpreter_name in interpreters:
            _ = self._get_interpretations(
                data, model, dataset_name, interpreter=interpreter_name)

  def _run_annotators(
      self, dataset: lit_dataset.Dataset) -> lit_dataset.Dataset:
    datapoints = [dict(ex) for ex in dataset.examples]
    annotated_spec = dict(dataset.spec())
    for annotator in self._annotators:
      annotator.annotate(datapoints, dataset, annotated_spec)
    return lit_dataset.Dataset(
        base=dataset, examples=datapoints, spec=annotated_spec)

  def make_handler(self, fn):
    """Convenience wrapper to handle args and serialization.

    This is a thin shim between server (handler, request, environ) and model
    logic (inputs, args, outputs).

    Args:
      fn: function (JsonDict, **kw) -> JsonDict

    Returns:
      fn wrapped as a request handler
    """

    @functools.wraps(fn)
    def _handler(app: wsgi_app.App, request, environ):
      kw = request.args.to_dict()
      # The frontend needs "simple" data (e.g. NumPy arrays converted to lists),
      # but for requests from Python we may want to use the invertible encoding
      # so that datatypes from remote models are the same as local ones.
      response_simple_json = utils.coerce_bool(
          kw.pop('response_simple_json', True))
      data = serialize.from_json(request.data) if len(request.data) else None
      # Special handling to dereference IDs.
      if data and 'inputs' in data.keys() and 'dataset_name' in kw:
        data['inputs'] = self._reconstitute_inputs(data['inputs'],
                                                   kw['dataset_name'])

      outputs = fn(data, **kw)
      response_body = serialize.to_json(outputs, simple=response_simple_json)
      return app.respond(request, response_body, 'application/json', 200)

    return _handler

  def __init__(
      self,
      models: Mapping[Text, lit_model.Model],
      datasets: Mapping[Text, lit_dataset.Dataset],
      generators: Optional[Mapping[Text, lit_components.Generator]] = None,
      interpreters: Optional[Mapping[Text, lit_components.Interpreter]] = None,
      annotators: Optional[List[lit_components.Annotator]] = None,
      layouts: Optional[dtypes.LitComponentLayouts] = None,
      # General server config; see server_flags.py.
      data_dir: Optional[Text] = None,
      warm_start: float = 0.0,
      warm_projections: bool = False,
      client_root: Optional[Text] = None,
      demo_mode: bool = False,
      default_layout: Optional[str] = None,
      canonical_url: Optional[str] = None,
      page_title: Optional[str] = None,
      development_demo: bool = False,
  ):
    if client_root is None:
      raise ValueError('client_root must be set on application')
    self._demo_mode = demo_mode
    self._development_demo = development_demo
    self._default_layout = default_layout
    self._canonical_url = canonical_url
    self._page_title = page_title
    self._data_dir = data_dir
    self._layouts = layouts or {}
    if data_dir and not os.path.isdir(data_dir):
      os.mkdir(data_dir)

    # Wrap models in caching wrapper
    self._models = {
        name: caching.CachingModelWrapper(model, name, cache_dir=data_dir)
        for name, model in models.items()
    }

    self._datasets = dict(datasets)
    self._datasets['_union_empty'] = lit_dataset.NoneDataset(self._models)

    self._annotators = annotators or []

    # Run annotation on each dataset, creating an annotated dataset and
    # replace the datasets with the annotated versions.
    for ds_key, ds in self._datasets.items():
      self._datasets[ds_key] = self._run_annotators(ds)

    # Index all datasets
    self._datasets = lit_dataset.IndexedDataset.index_all(
        self._datasets, caching.input_hash)

    if generators is not None:
      self._generators = generators
    else:
      self._generators = {
          'Ablation Flip': ablation_flip.AblationFlip(),
          'Hotflip': hotflip.HotFlip(),
          'Scrambler': scrambler.Scrambler(),
          'Word Replacer': word_replacer.WordReplacer(),
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
          'Grad L2 Norm': gradient_maps.GradientNorm(),
          'Grad ⋅ Input': gradient_maps.GradientDotInput(),
          'Integrated Gradients': gradient_maps.IntegratedGradients(),
          'LIME': lime_explainer.LIME(),
          'Model-provided salience': model_salience.ModelSalience(self._models),
          'counterfactual explainer': lemon_explainer.LEMON(),
          'tcav': tcav.TCAV(),
          'thresholder': thresholder.Thresholder(),
          'nearest neighbors': nearest_neighbors.NearestNeighbors(),
          'metrics': metrics_group,
          'pdp': pdp.PdpInterpreter(),
          # Embedding projectors expose a standard interface, but get special
          # handling so we can precompute the projections if requested.
          'pca': projection.ProjectionManager(pca.PCAModel),
          'umap': projection.ProjectionManager(umap.UmapModel),
      }

    # Information on models, datasets, and other components.
    self._info = self._build_metadata()

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
        '/create_dataset': self._create_dataset,
        '/create_model': self._create_model,
        '/get_generated': self._get_generated,
        '/save_datapoints': self._save_datapoints,
        '/load_datapoints': self._load_datapoints,
        '/annotate_new_data': self._annotate_new_data,
        # Model prediction endpoints.
        '/get_preds': self._get_preds,
        '/get_interpretations': self._get_interpretations,
    }

    self._wsgi_app = wsgi_app.App(
        # Wrap endpoint fns to take (handler, request, environ)
        handlers={k: self.make_handler(v) for k, v in handlers.items()},
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
