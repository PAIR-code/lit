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
"""LIT backend, as a standard WSGI app."""

import functools
import glob
import math
import os
import random
import threading
import time
from typing import Optional, Mapping, Sequence, Union, Callable, Iterable

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import layout
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import core
from lit_nlp.lib import caching
from lit_nlp.lib import flag_helpers
from lit_nlp.lib import serialize
from lit_nlp.lib import ui_state
from lit_nlp.lib import utils
from lit_nlp.lib import validation
from lit_nlp.lib import wsgi_app
import tqdm

JsonDict = types.JsonDict
Input = types.Input
IndexedInput = types.IndexedInput

# Export this symbol, for access from demo.py
PredsCache = caching.PredsCache

ProgressIndicator = Callable[[Iterable], Iterable]


class LitApp(object):
  """LIT WSGI application."""

  def _build_metadata(self):
    """Build metadata from model and dataset specs."""
    model_info = {}
    for name, m in self._models.items():
      mspec: lit_model.ModelSpec = m.spec()
      info = {
          'description': m.description(),
          'spec': {
              'input': mspec.input,
              'output': mspec.output
          }
      }

      # List compatible datasets.
      info['datasets'] = [
          name for name, dataset in self._datasets.items()
          if mspec.is_compatible_with_dataset(dataset.spec())
      ]
      if len(info['datasets']) == 0:  # pylint: disable=g-explicit-length-test
        logging.error("Error: model '%s' has no compatible datasets!", name)

      compat_gens: set[str] = set()
      compat_interps: set[str] = set()

      for d in info['datasets']:
        dataset: lit_dataset.Dataset = self._datasets[d]
        compat_gens.update([
            name for name, gen in self._generators.items()
            if gen.is_compatible(model=m, dataset=dataset)
        ])
        compat_interps.update([
            name for name, interp in self._interpreters.items()
            if interp.is_compatible(model=m, dataset=dataset)
        ])

      info['generators'] = [name for name in self._generators.keys()
                            if name in compat_gens]
      info['interpreters'] = [name for name in self._interpreters.keys()
                              if name in compat_interps]
      model_info[name] = info

    dataset_info = {}
    for name, ds in self._datasets.items():
      dataset_info[name] = {
          'spec': ds.spec(),
          'description': ds.description(),
          'size': len(ds),
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
        'inlineDoc': self._inline_doc,
        'onboardStartDoc': self._onboard_start_doc,
        'onboardEndDoc': self._onboard_end_doc,
        'syncState': self.ui_state_tracker is not None,
    }

  def _get_model_spec(self, name: str):
    return self._info['models'][name]['spec']

  def _get_info(self, unused_data, **unused_kw):
    """Get model info and send to frontend."""
    return self._info

  def _reconstitute_inputs(self, inputs: Sequence[Union[IndexedInput, str]],
                           dataset_name: str) -> list[IndexedInput]:
    """Reconstitute any inputs sent as references (bare IDs)."""
    index = self._datasets[dataset_name].index
    # TODO(b/178228238): set up proper debug logging and hide this by default.
    num_aliased = sum([isinstance(ex, str) for ex in inputs])
    logging.info(
        "%d of %d inputs sent as IDs; reconstituting from dataset '%s'",
        num_aliased, len(inputs), dataset_name)
    return [index[ex] if isinstance(ex, str) else ex for ex in inputs]

  def _save_datapoints(self, data, dataset_name: str, path: str, **unused_kw):
    """Save datapoints to disk."""
    if self._demo_mode:
      logging.warning('Attempted to save datapoints in demo mode.')
      return None
    return self._datasets[dataset_name].save(data['inputs'], path)

  def _load_datapoints(self, unused_data, dataset_name: str, path: str,
                       **unused_kw):
    """Load datapoints from disk."""
    if self._demo_mode:
      logging.warning('Attempted to load datapoints in demo mode.')
      return None
    dataset = self._datasets[dataset_name].load(path)
    return dataset.indexed_examples

  def _get_preds(self,
                 data,
                 model: str,
                 dataset_name: Optional[str] = None,
                 requested_types: Optional[str] = None,
                 requested_fields: Optional[str] = None,
                 **kw):
    """Get model predictions.

    Args:
      data: data payload, containing 'inputs' field
      model: name of the model to run
      dataset_name: name of the active dataset
      requested_types: optional, comma-separated list of type names to return
      requested_fields: optional, comma-separated list of field names to return
        in addition to the ones returned due to 'requested_types'.
      **kw: additional args passed to model.predict_with_metadata()

    Returns:
      list[JsonDict] containing requested fields of model predictions
    """
    preds = list(self._models[model].predict_with_metadata(
        data['inputs'], dataset_name=dataset_name, **kw))
    if not requested_types and not requested_fields:
      return preds

    # Figure out what to return to the frontend.
    output_spec = self._get_model_spec(model)['output']
    requested_types = requested_types.split(',') if requested_types else []
    requested_fields = requested_fields.split(',') if requested_fields else []
    logging.info('Requested types: %s, fields: %s', str(requested_types),
                 str(requested_fields))
    for t_name in requested_types:
      t_class = getattr(types, t_name, None)
      if not issubclass(t_class, types.LitType):
        raise TypeError(f"Class '{t_name}' is not a valid LitType.")
      requested_fields.extend(utils.find_spec_keys(output_spec, t_class))
    ret_keys = set(requested_fields)  # de-dupe

    # Return selected keys.
    logging.info('Will return keys: %s', str(ret_keys))
    # One record per input.
    ret = [utils.filter_by_keys(p, ret_keys.__contains__) for p in preds]
    return ret

  def _annotate_new_data(self,
                         data,
                         dataset_name: Optional[str] = None,
                         **unused_kw) -> list[IndexedInput]:
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

  def _post_new_data(
      self, data, dataset_name: Optional[str] = None,
      **unused_kw) -> dict[str, str]:
    """Save datapoints provided, after annotatation, for later retrieval.

    Args:
      data: JsonDict of datapoints to add, in dict under key 'inputs', per
        format for other requests.
      dataset_name: Dataset containing the format of data to add, necessary for
        proper datapoint annotation.

    Returns:
      A dict of two URLs (minus the root of the webserver). 'load' value is
      for loading LIT with those datapoints. 'remove' value is for removing
      those new datapoints from this server after they have been loaded, if
      desired.
    """
    assert 'inputs' in data, 'Data dict does not contain "inputs" field'
    data_with_metadata = [
        {'data': d,
         'meta': {'added': True, 'source': 'POST', 'parentId': None}}
        for d in data['inputs']]
    annotation_input = {'inputs': data_with_metadata}
    annotated_data = self._annotate_new_data(annotation_input, dataset_name)
    datapoints_id = utils.get_uuid()
    with self._saved_datapoints_lock:
      self._saved_datapoints[datapoints_id] = annotated_data
    return {
        'load': f'?saved_datapoints_id={datapoints_id}',
        'remove': f'/remove_new_data?saved_datapoints_id={datapoints_id}'}

  def _fetch_new_data(self, unused_data, saved_datapoints_id: str, **unused_kw):
    with self._saved_datapoints_lock:
      assert saved_datapoints_id in self._saved_datapoints, (
          'No saved data with ID %s' % saved_datapoints_id)
      return self._saved_datapoints[saved_datapoints_id]

  def _remove_new_data(
      self, unused_data, saved_datapoints_id: str, **unused_kw):
    with self._saved_datapoints_lock:
      assert saved_datapoints_id in self._saved_datapoints, (
          'No saved data with ID %s' % saved_datapoints_id)
      del self._saved_datapoints[saved_datapoints_id]

  def _get_dataset(self,
                   unused_data,
                   dataset_name: Optional[str] = None,
                   **unused_kw) -> list[IndexedInput]:
    """Attempt to get dataset, or override with a specific path."""
    return self._datasets[dataset_name].indexed_examples

  def _create_dataset(self,
                      unused_data,
                      dataset_name: Optional[str] = None,
                      dataset_path: Optional[str] = None,
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
      logging.error('Not able to load: %s', dataset_name)
      return None

  def _create_model(self,
                    unused_data,
                    model_name: Optional[str] = None,
                    model_path: Optional[str] = None,
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

  def _get_generated(self, data, model: str, dataset_name: str, generator: str,
                     **unused_kw):
    """Generate new datapoints based on the request."""
    generator_name = generator
    generator: lit_components.Generator = self._generators[generator_name]
    dataset = self._datasets[dataset_name]
    # Nested list, containing generated examples from each input.
    all_generated: list[list[Input]] = generator.run_with_metadata(
        data['inputs'], self._models[model], dataset, config=data.get('config'))

    # Annotate datapoints
    def annotate_generated(datapoints):
      dataset_to_annotate = lit_dataset.Dataset(
          base=dataset, examples=datapoints)
      annotated_dataset = self._run_annotators(dataset_to_annotate)
      return annotated_dataset.examples

    annotated_generated = [
        annotate_generated(generated) for generated in all_generated
    ]

    # Add metadata.
    all_generated_indexed: list[list[IndexedInput]] = [
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

  def _get_interpretations(self, data, model: str, dataset_name: str,
                           interpreter: str, **unused_kw):
    """Run an interpretation component."""
    interpreter = self._interpreters[interpreter]
    # Get model preds before the interpreter call. Usually these are cached.
    # TODO(lit-dev): see if we can remove this path and just allow interpreters
    # to call the model directly.
    if model:
      assert model in self._models, f"Model '{model}' is not a valid model."
      model_outputs = self._get_preds(data, model, dataset_name)
      model = self._models[model]
    else:
      model_outputs = None
      model = None

    return interpreter.run_with_metadata(
        data['inputs'],
        model,
        self._datasets[dataset_name],
        model_outputs=model_outputs,
        config=data.get('config'))

  def _push_ui_state(self, data, dataset_name: str, **unused_kw):
    """Push UI state back to Python."""
    if self.ui_state_tracker is None:
      raise RuntimeError('Attempted to push UI state, but that is not enabled '
                         'for this server.')
    options = data.get('config', {})
    self.ui_state_tracker.update_state(data['inputs'],
                                       self._datasets[dataset_name],
                                       dataset_name, **options)

  def _validate(self, validate: Optional[flag_helpers.ValidationMode],
                report_all: bool):
    """Validate all datasets and models loaded for proper setup."""
    if validate is None or validate == flag_helpers.ValidationMode.OFF:
      return

    datasets_to_validate = {}
    for dataset in self._datasets:
      if validate == flag_helpers.ValidationMode.ALL:
        datasets_to_validate[dataset] = self._datasets[dataset]
      elif validate == flag_helpers.ValidationMode.FIRST:
        datasets_to_validate[dataset] = self._datasets[dataset].slice[:1]
      elif validate == flag_helpers.ValidationMode.SAMPLE:
        sample_size = math.ceil(len(self._datasets[dataset]) * 0.05)
        datasets_to_validate[dataset] = self._datasets[dataset].sample(
            sample_size)
    for dataset in datasets_to_validate:
      logging.info("Validating dataset '%s'", dataset)
      validation.validate_dataset(
          datasets_to_validate[dataset], report_all)
    for model, model_info in self._info['models'].items():
      for dataset_name in model_info['datasets']:
        logging.info("Validating model '%s' on dataset '%s'", model,
                     dataset_name)
        validation.validate_model(
            self._models[model], datasets_to_validate[dataset_name], report_all)

  def _warm_start(self,
                  rate: float,
                  progress_indicator: Optional[ProgressIndicator] = None):
    """Warm-up the predictions cache by making some model calls."""
    assert rate >= 0 and rate <= 1
    for model, model_info in self._info['models'].items():
      for dataset_name in model_info['datasets']:
        logging.info("Warm-start of model '%s' on dataset '%s'", model,
                     dataset_name)
        all_examples: list[IndexedInput] = self._get_dataset([], dataset_name)
        if rate < 1:
          examples = random.sample(all_examples, int(len(all_examples) * rate))
          logging.info('Partial warm-start: running on %d/%d examples.',
                       len(examples), len(all_examples))
        else:
          examples = all_examples
        _ = self._get_preds({'inputs': examples},
                            model,
                            dataset_name,
                            progress_indicator=progress_indicator)

  def _warm_projections(self, interpreters: list[str]):
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
              use_input=False,
              proj_kw={'n_components': 3})
          data = {'inputs': [], 'config': config}
          for interpreter_name in interpreters:
            _ = self._get_interpretations(
                data, model, dataset_name, interpreter=interpreter_name)

  def _run_annotators(self,
                      dataset: lit_dataset.Dataset) -> lit_dataset.Dataset:
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
      models: Mapping[str, lit_model.Model],
      datasets: Mapping[str, lit_dataset.Dataset],
      generators: Optional[Mapping[str, lit_components.Generator]] = None,
      interpreters: Optional[Mapping[str, lit_components.Interpreter]] = None,
      annotators: Optional[list[lit_components.Annotator]] = None,
      layouts: Optional[layout.LitComponentLayouts] = None,
      # General server config; see server_flags.py.
      data_dir: Optional[str] = None,
      warm_start: float = 0.0,
      warm_start_progress_indicator: Optional[ProgressIndicator] = tqdm
      .tqdm,  # not in server_flags
      warm_projections: bool = False,
      client_root: Optional[str] = None,
      demo_mode: bool = False,
      default_layout: Optional[str] = None,
      canonical_url: Optional[str] = None,
      page_title: Optional[str] = None,
      development_demo: bool = False,
      inline_doc: Optional[str] = None,
      onboard_start_doc: Optional[str] = None,
      onboard_end_doc: Optional[str] = None,
      sync_state: bool = False,  # notebook-only; not in server_flags
      validate: Optional[flag_helpers.ValidationMode] = None,
      report_all: bool = False,
  ):
    if client_root is None:
      raise ValueError('client_root must be set on application')
    self._demo_mode = demo_mode
    self._development_demo = development_demo
    self._default_layout = default_layout
    self._canonical_url = canonical_url
    self._page_title = page_title
    self._inline_doc = inline_doc
    self._onboard_start_doc = onboard_start_doc
    self._onboard_end_doc = onboard_end_doc
    self._data_dir = data_dir
    if data_dir and not os.path.isdir(data_dir):
      os.mkdir(data_dir)

    # TODO(lit-dev): override layouts instead of merging, to allow clients
    # to opt-out of the default bundled layouts. This will require updating
    # client code to manually merge when this is the desired behavior.
    self._layouts = dict(layout.DEFAULT_LAYOUTS, **(layouts or {}))

    # Wrap models in caching wrapper
    self._models = {
        name: caching.CachingModelWrapper(model, name, cache_dir=data_dir)
        for name, model in models.items()
    }

    self._datasets: dict[str, lit_dataset.Dataset] = dict(datasets)
    # TODO(b/202210900): get rid of this, just dynamically create the empty
    # dataset on the frontend.
    self._datasets['_union_empty'] = lit_dataset.NoneDataset(self._models)

    self._annotators = annotators or []

    self._saved_datapoints = {}
    self._saved_datapoints_lock = threading.Lock()

    # Run annotation on each dataset, creating an annotated dataset and
    # replace the datasets with the annotated versions.
    for ds_key, ds in self._datasets.items():
      self._datasets[ds_key] = self._run_annotators(ds)

    # Index all datasets
    self._datasets = lit_dataset.IndexedDataset.index_all(
        self._datasets, caching.input_hash)

    # Generator initialization
    if generators is not None:
      self._generators = generators
    else:
      self._generators = core.default_generators()

    # Interpreter initialization
    if interpreters is not None:
      self._interpreters = interpreters
    else:
      self._interpreters = core.default_interpreters(self._models)

    # Component to sync state from TS -> Python. Used in notebooks.
    if sync_state:
      self.ui_state_tracker = ui_state.UIStateTracker()
    else:
      self.ui_state_tracker = None

    # Information on models, datasets, and other components.
    self._info = self._build_metadata()

    # Validate datasets and models if specified.
    self._validate(validate, report_all)

    # Optionally, run models to pre-populate cache.
    if warm_projections:
      logging.info(
          'Projection (dimensionality reduction) warm-start requested; '
          'will do full warm-start for all models since predictions are needed.'
      )
      warm_start = 1.0

    if warm_start > 0:
      self._warm_start(
          rate=warm_start, progress_indicator=warm_start_progress_indicator)
      self.save_cache()
      if warm_start >= 1:
        warm_projections = True

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
        '/post_new_data': self._post_new_data,
        '/fetch_new_data': self._fetch_new_data,
        '/remove_new_data': self._remove_new_data,
        '/push_ui_state': self._push_ui_state,
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
