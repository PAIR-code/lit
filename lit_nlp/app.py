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

from collections.abc import Callable, Iterable, Mapping, Sequence
import functools
import glob
import math
import os
import random
import threading
import time
from typing import Any, Optional, TypedDict, Union

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

DatasetLoader = tuple[Callable[..., lit_dataset.Dataset], Optional[types.Spec]]
DatasetLoadersMap = dict[str, DatasetLoader]

ModelLoader = tuple[Callable[..., lit_model.Model], Optional[types.Spec]]
ModelLoadersMap = dict[str, ModelLoader]

_EMPTY_DATASET_KEY = '_union_empty'


# LINT.IfChange
class ComponentInfo(TypedDict):
  configSpec: types.Spec  # pylint: disable=invalid-name  # Named for JSON struct
  metaSpec: types.Spec    # pylint: disable=invalid-name  # Named for JSON struct
  description: str
# LINT.ThenChange(./client/lib/types.ts)


def _get_component_info(
    obj: lit_components.Interpreter,
) -> ComponentInfo:
  """Returns the ComponentInfo for an Interpreter, Generator, Metric, etc."""
  return ComponentInfo(
      configSpec=obj.config_spec(),
      metaSpec=obj.meta_spec(),
      description=obj.description(),
  )


def _get_compatible_names(
    candidates: Mapping[str, lit_components.Interpreter],
    model: lit_model.Model,
    dataset: lit_dataset.Dataset,
) -> Sequence[str]:
  """Returns the names of the candidates compatible with the model/dataset."""
  return [
      name
      for name, candidate in candidates.items()
      if candidate.is_compatible(model=model, dataset=dataset)
  ]


class LitApp(object):
  """LIT WSGI application."""

  def _build_metadata(self):
    """Build metadata from model and dataset specs."""
    model_info = {}
    for name, model in self._models.items():
      info = {
          'description': model.description(),
          'spec': {
              'input': model.input_spec(),
              'output': model.output_spec(),
          }
      }

      # List compatible datasets.
      info['datasets'] = [
          name for name, dataset in self._datasets.items()
          if model.is_compatible_with_dataset(dataset)
      ]
      if len(info['datasets']) == 0:  # pylint: disable=g-explicit-length-test
        logging.error("Error: model '%s' has no compatible datasets!", name)

      compat_gens: set[str] = set()
      compat_interps: set[str] = set()
      compat_metrics: set[str] = set()

      for d in info['datasets']:
        dataset: lit_dataset.Dataset = self._datasets[d]
        compat_gens.update(
            _get_compatible_names(self._generators, model, dataset)
        )
        compat_interps.update(
            _get_compatible_names(self._interpreters, model, dataset)
        )
        compat_metrics.update(
            _get_compatible_names(self._metrics, model, dataset)
        )

      info['generators'] = [
          name for name in self._generators.keys() if name in compat_gens
      ]
      info['interpreters'] = [
          name for name in self._interpreters.keys() if name in compat_interps
      ]
      info['metrics'] = [
          name for name in self._metrics.keys() if name in compat_metrics
      ]
      model_info[name] = info

    dataset_info = {}
    for name, ds in self._datasets.items():
      dataset_info[name] = {
          'spec': ds.spec(),
          'description': ds.description(),
          'size': len(ds),
      }

    generator_info: Mapping[str, ComponentInfo] = {
        name: _get_component_info(gen) for name, gen in self._generators.items()
    }

    interpreter_info: Mapping[str, ComponentInfo] = {
        name: _get_component_info(interp)
        for name, interp in self._interpreters.items()
    }

    metrics_info: Mapping[str, ComponentInfo] = {
        name: _get_component_info(metric)
        for name, metric in self._metrics.items()
    }

    init_specs = {
        'datasets': {n: s for n, (_, s) in self._dataset_loaders.items()},
        'models': {n: s for n, (_, s) in self._model_loaders.items()},
    }

    return {
        # Component info and specs
        'models': model_info,
        'datasets': dataset_info,
        'generators': generator_info,
        'interpreters': interpreter_info,
        'metrics': metrics_info,
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
        'initSpecs': init_specs,
    }

  def _get_model_spec(self, name: str):
    return self._info['models'][name]['spec']

  def _get_info(self, unused_data, **unused_kw):
    """Get model info and send to frontend."""
    return self._info

  def _reconstitute_inputs(
      self, inputs: Sequence[Union[IndexedInput, str]], dataset_name: str
  ) -> list[IndexedInput]:
    """Reconstitute any inputs sent as references (bare IDs)."""
    index = self._datasets[dataset_name].index
    # TODO(b/178228238): set up proper debug logging and hide this by default.
    # TODO(b/171513556): Reconsistute as Inputs instead of IndexedInputs
    num_aliased = sum([isinstance(ex, str) for ex in inputs])
    logging.info(
        "%d of %d inputs sent as IDs; reconstituting from dataset '%s'",
        num_aliased,
        len(inputs),
        dataset_name,
    )
    return [index[ex] if isinstance(ex, str) else ex for ex in inputs]

  def _save_datapoints(
      self,
      data,
      dataset_name: Optional[str] = None,
      path: Optional[str] = None,
      **unused_kw,
  ):
    """Save datapoints to disk."""
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to save datapoints.')
    if path is None:
      raise ValueError('Must provide a "path" to save datapoints.')

    if self._demo_mode:
      logging.warning('Attempted to save datapoints in demo mode.')
      return None
    return self._datasets[dataset_name].save(data['inputs'], path)

  def _load_datapoints(
      self,
      unused_data,
      dataset_name: Optional[str] = None,
      path: Optional[str] = None,
      **unused_kw,
  ):
    """Load datapoints from disk."""
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to load datapoints.')
    if path is None:
      raise ValueError('Must provide a "path" from which to load datapoints.')

    if self._demo_mode:
      logging.warning('Attempted to load datapoints in demo mode.')
      return None
    dataset = self._datasets[dataset_name].load(path)
    return dataset.indexed_examples

  def _get_preds(self,
                 data: types.JsonDict,
                 model: Optional[str] = None,
                 requested_types: Optional[str] = None,
                 requested_fields: Optional[str] = None,
                 **kw):
    """Get model predictions.

    Args:
      data: data payload, containing 'inputs' field
      model: name of the model to run
      requested_types: optional, comma-separated list of type names to return
      requested_fields: optional, comma-separated list of field names to return
        in addition to the ones returned due to 'requested_types'.
      **kw: additional args passed to model.predict()

    Returns:
      list[JsonDict] containing requested fields of model predictions

    Raises:
      KeyError: If `data` does not have an 'inputs' property.
      TypeError: If one of entries in `requested_types` is not a valid LitType.
      ValueError: If the model returns a different number of predictions than
        the number of inputs.
    """
    if model is None:
      raise ValueError('Must provide a "model" name to get preds from.')

    inputs = data['inputs']
    preds = list(self._models[model].predict(
        [ex['data'] for ex in inputs], **kw))

    num_preds = len(preds)
    num_inputs = len(inputs)
    if num_preds != num_inputs:
      raise ValueError(
          f'Different number of model predictions ({num_preds}) than inputs'
          f' ({num_inputs}).'
      )

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
                         data: types.JsonDict,
                         dataset_name: Optional[str] = None,
                         **unused_kw) -> list[IndexedInput]:
    """Fill in index and other extra data for the provided datapoints."""
    # TODO(lit-dev): unify this with hash fn on dataset objects.
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to annotate.')

    # Generate annotated versions of new datapoints.
    dataset = self._datasets[dataset_name]
    input_examples = [example['data'] for example in data['inputs']]
    dataset_to_annotate = lit_dataset.Dataset(
        base=dataset, examples=input_examples)
    annotated_dataset = self._run_annotators(dataset_to_annotate)

    # Add annotations and IDs to new datapoints.
    for i, example in enumerate(data['inputs']):
      new_id = caching.input_hash(example['data'])
      example['data'] = dict(annotated_dataset.examples[i], _id=new_id)
      example['id'] = new_id

    return data['inputs']  # pytype: disable=bad-return-type  # always-use-return-annotations

  def _post_new_data(
      self,
      data: types.JsonDict,
      dataset_name: Optional[str] = None,
      **unused_kw
  ) -> dict[str, str]:
    """Save datapoints provided, after annotatation, for later retrieval.

    Args:
      data: JsonDict of datapoints to add, in dict under key 'inputs', per
        format for other requests.
      dataset_name: Dataset containing the format of data to add, necessary for
        proper datapoint annotation.

    Returns:
      A dict of two URLs (minus the root of the webserver). 'load' value is
      for loading LIT with those datapoints. 'remove' value is for removing
      those new datapoints from this server after they have been loaded, if
      desired.

    Raises:
      KeyError: If the `data` dictionary does not have an "inputs" field.
      ValueError: If a "dataset_name" is not provided.
    """
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to save new datapoints.')
    if 'inputs' not in data:
      raise KeyError('Data dict does not contain "inputs" field.')

    data_with_metadata = [
        {'data': d, 'meta': {'added': True, 'source': 'POST', 'parentId': None}}
        for d in data['inputs']
    ]
    annotation_input = {'inputs': data_with_metadata}
    annotated_data = self._annotate_new_data(annotation_input, dataset_name)
    datapoints_id = utils.get_uuid()
    with self._saved_datapoints_lock:
      self._saved_datapoints[datapoints_id] = annotated_data
    return {
        'load': f'?saved_datapoints_id={datapoints_id}',
        'remove': f'/remove_new_data?saved_datapoints_id={datapoints_id}',
    }

  def _fetch_new_data(
      self, unused_data, saved_datapoints_id: Optional[str] = None, **unused_kw
  ):
    if not saved_datapoints_id:
      raise ValueError('Must provide a "saved_datapoints_id" to get data from.')

    with self._saved_datapoints_lock:
      if saved_datapoints_id not in self._saved_datapoints:
        raise ValueError(f'No saved data with ID: {saved_datapoints_id}')
      return self._saved_datapoints[saved_datapoints_id]

  def _remove_new_data(
      self, unused_data, saved_datapoints_id: Optional[str] = None, **unused_kw
  ):
    if not saved_datapoints_id:
      raise ValueError('Must provide a "saved_datapoints_id" to remove data.')

    with self._saved_datapoints_lock:
      if saved_datapoints_id not in self._saved_datapoints:
        raise ValueError(f'No saved data with ID: {saved_datapoints_id}')
      del self._saved_datapoints[saved_datapoints_id]

  def _get_dataset(
      self, unused_data, dataset_name: Optional[str] = None, **unused_kw
  ) -> list[IndexedInput]:
    """Attempt to get dataset, or override with a specific path."""
    if not dataset_name:
      raise ValueError('Must provide a "dataset_name" to get examples.')
    return list(self._datasets[dataset_name].indexed_examples)

  def _create_dataset(
      self,
      data: types.JsonDict,
      dataset_name: Optional[str] = None,
      **unused_kw,
  ):
    """Create a dataset, updating and returning the metadata."""
    if dataset_name is None:
      raise ValueError('No base dataset specified.')

    config: Optional[dict[str, Any]] = data.get('config')
    if config is None:
      raise ValueError('No config specified.')

    new_name: Optional[str] = config.pop('new_name', None)
    if new_name is None:
      raise ValueError('No name provided for the new dataset.')
    elif new_name in self._datasets:
      return (self._info, new_name)  # Return the existing dataset

    if (loader_info := self._dataset_loaders.get(dataset_name)) is None:
      raise ValueError(
          f'No loader information (Cls + init_spec) found for {dataset_name}'
      )

    dataset_cls, dataset_init_spec = loader_info

    if dataset_init_spec is not None:
      utils.validate_config_against_spec(
          config,
          dataset_init_spec,
          f'{dataset_name} ({dataset_cls.__name__})',
          raise_for_unsupported=True,
      )

    new_dataset = dataset_cls(**config)
    annotated_dataset = self._run_annotators(new_dataset)
    self._datasets[new_name] = lit_dataset.IndexedDataset(
        base=annotated_dataset, id_fn=caching.input_hash
    )
    self._info = self._build_metadata()
    return (self._info, new_name)

  def _create_model(self,
                    data: types.JsonDict,
                    model_name: Optional[str] = None,
                    **unused_kw):
    """Create a model, updating and returning the metadata."""
    if model_name is None:
      raise ValueError('No base model specified.')

    config: Optional[dict[str, Any]] = data.get('config')
    if config is None:
      raise ValueError('No config specified.')

    new_name: Optional[str] = config.pop('new_name', None)
    if new_name is None:
      raise ValueError('No name provided for the new model.')
    elif new_name in self._models:
      return (self._info, new_name)  # Return the existing model

    if (loader_info := self._model_loaders.get(model_name)) is None:
      raise ValueError(
          f'No loader information (Cls + init_spec) found for {model_name}'
      )

    model_cls, model_init_spec = loader_info

    if model_init_spec is not None:
      utils.validate_config_against_spec(
          config,
          model_init_spec,
          f'{model_name} ({model_cls.__name__})',
          raise_for_unsupported=True,
      )

    new_model = model_cls(**config)
    self._models[new_name] = caching.CachingModelWrapper(
        new_model, new_name, **self._caching_model_wrapper_kw
    )
    empty_dataset = lit_dataset.NoneDataset(self._models)
    self._datasets[_EMPTY_DATASET_KEY] = lit_dataset.IndexedDataset(
        base=self._run_annotators(empty_dataset), id_fn=caching.input_hash
    )
    self._info = self._build_metadata()
    return (self._info, new_name)

  def _get_generated(
      self,
      data: types.JsonDict,
      model: Optional[str] = None,
      dataset_name: Optional[str] = None,
      generator: Optional[str] = None,
      **unused_kw,
  ):
    """Generate new datapoints based on the request."""
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to get base examples.')
    if generator is None:
      raise ValueError('Must provide a "generator" name to generate examples.')
    if model is None:
      raise ValueError('Must provide a "model" name to get predictions.')

    genny: lit_components.Generator = self._generators[generator]
    config_spec: types.Spec = genny.config_spec()
    config: Optional[types.JsonDict] = data.get('config')

    if config_spec and config is not None:
      utils.validate_config_against_spec(
          config, config_spec, f'{generator} ({type(genny).__name__})'
      )

    dataset = self._datasets[dataset_name]
    # Nested list, containing generated examples from each input.
    all_generated: list[list[Input]] = genny.run(  # pytype: disable=annotation-type-mismatch  # always-use-return-annotations
        [ex['data'] for ex in data['inputs']],
        self._models[model],
        dataset,
        config=config)

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
            'source': generator,
            'added': True,
        })
    return all_generated_indexed

  def _get_interpretations(
      self,
      data: types.JsonDict,
      model: Optional[str] = None,
      dataset_name: Optional[str] = None,
      interpreter: Optional[str] = None,
      # boolean but via URL param, so encoding as "0" /  "1" is safer.
      do_predict: str = '1',
      **unused_kw,
  ):
    """Run an interpretation component."""
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to get examples.')
    if interpreter is None:
      raise ValueError('Must provide a "interpreter" name to interpret preds.')
    if model is None:
      raise ValueError('Must provide a "model" name to get predictions.')

    interp: lit_components.Interpreter = self._interpreters[interpreter]
    mdl: lit_model.Model = self._models[model]

    config_spec: types.Spec = interp.config_spec()
    config: Optional[types.JsonDict] = data.get('config')
    if config_spec and config is not None:
      utils.validate_config_against_spec(
          config, config_spec, f'{interpreter} ({type(interp).__name__})'
      )

    model_inputs = [ex['data'] for ex in data['inputs']]

    # Get model preds before the interpreter call. Usually these are cached.
    # TODO(b/278586715): See if we can remove this path and just allow
    # interpreters to call the model directly.
    if utils.coerce_bool(do_predict):
      # Workaround so that interpreters can skip the predict() call when it
      # is unnecessary and may be slow.
      # TODO(b/278586715): Remove this once we can ensure that model_outputs
      # can be removed from the Interpreter API.
      model_outputs = list(mdl.predict(model_inputs))
      assert len(model_outputs) == len(model_inputs)
    else:
      model_outputs = None

    return interp.run(
        model_inputs,
        mdl,
        self._datasets[dataset_name],
        model_outputs=model_outputs,
        config=data.get('config'),
    )

  def _get_metrics(
      self,
      data: types.JsonDict,
      model: Optional[str] = None,
      dataset_name: Optional[str] = None,
      metrics: Optional[str] = None,
      # TODO(b/278586715): Remove this parameter once linked bug is fixed.
      do_predict: str = '1',  # bool URL param; encoding as "0" /  "1" is safer.
      **unused_kw,
  ) -> types.JsonDict:
    """Run the specified Metrics components.

    Args:
      data: JSON parsed from the HTTP Request body containing the inputs
        (required) and config (optional) for parameterizing the Metrics calls.
      model: The name of the model loaded in LIT, used to fetch the model
        predictions.
      dataset_name: The name of the dataset containing the ground truth labels
        for the provided inputs.
      metrics: An optional comma-separated string of metrics to run, if None it
        will run all Metrics loaded in this LitApp instance.
      do_predict: If true (default), will fetch the model predictions in this
        function using `_get_preds()` and pass them through to each Metrics
        component's run function.
      **unused_kw: Unused keyword arguments.

    Returns:
      A dictionary of metrics results where the keys are the name of the
      Metrics component and the values are list of dictionaries containing the
      prediction key (`pred_key`), the label key (`label_key`), and `metrics`
      for that pair of keys as a `Mapping[str, float]`.

    Raises:
      KeyError: If a model, dataset, or metric with the specified name is not
        loaded in the LitApp instance.
      ValueError: If there are no inputs.
    """
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to get examples.')
    if model is None:
      raise ValueError('Must provide a "model" name to get predictions.')

    inputs = data.get('inputs')
    if not inputs:
      raise ValueError('Metrics cannot be computed without inputs.')

    if metrics:
      metrics_to_run = tuple(m for m in metrics.split(',') if m)
      unknown_metrics = [m for m in metrics_to_run if m not in self._metrics]
      if unknown_metrics:
        raise KeyError(f'Requested unknown metrics "{unknown_metrics}".')
    else:
      metrics_to_run = tuple(self._metrics.keys())

    if utils.coerce_bool(do_predict):
      model_outputs = self._get_preds(data=data, model=model)
    else:
      model_outputs = None

    dataset: lit_dataset.IndexedDataset = self._datasets[dataset_name]
    mdl: lit_model.Model = self._models[model]
    config: Optional[types.JsonDict] = data.get('config')

    results: dict[str, Any] = {}
    for name in metrics_to_run:
      # TODO(b/254833485): Add type annotation once the metrics wrapper classes
      # inherit from lit_component.Metrics.
      metric = self._metrics[name]

      config_spec: types.Spec = metric.config_spec()
      if config_spec and config is not None:
        utils.validate_config_against_spec(
            config, config_spec, f'Metric {name}'
        )

      results[name] = metric.run(
          [ex['data'] for ex in inputs],
          mdl,
          dataset,
          model_outputs=model_outputs,
          config=config
      )

    return results

  def _push_ui_state(
      self,
      data: types.JsonDict,
      dataset_name: Optional[str] = None,
      **unused_kw,
  ):
    """Push UI state back to Python."""
    tracker: Optional[ui_state.UIStateTracker] = self.ui_state_tracker
    if tracker is None:
      raise RuntimeError(
          'Attempted to push UI state, but that is not enabled for this server.'
      )
    if dataset_name is None:
      raise ValueError('Must provide a "dataset_name" to get base examples.')

    options = data.get('config', {})
    tracker.update_state(
        data['inputs'], self._datasets[dataset_name], dataset_name, **options
    )

  def _validate(
      self,
      validate: Optional[flag_helpers.ValidationMode],
      enforce_dataset_fields_required: bool = False,
      report_all: bool = False,
  ):
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
          datasets_to_validate[dataset],
          enforce_all_fields_required=enforce_dataset_fields_required,
          report_all=report_all
      )

    for model, model_info in self._info['models'].items():
      for dataset in model_info['datasets']:
        logging.info("Validating model '%s' on dataset '%s'", model, dataset)
        validation.validate_model(
            self._models[model], datasets_to_validate[dataset], report_all)

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
        _ = self._get_preds(data={'inputs': examples},
                            model=model,
                            progress_indicator=progress_indicator)

  def _warm_projections(self, interpreters: list[str]):
    """Pre-compute UMAP/PCA projections with default arguments."""
    for interpreter_name in interpreters:
      if interpreter_name not in self._interpreters:
        continue

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
            _ = self._get_interpretations(
                data=data,
                model=model,
                dataset_name=dataset_name,
                interpreter=interpreter_name)

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
          kw.pop('response_simple_json', True)
      )
      data = serialize.from_json(request.data) if len(request.data) else None
      # Special handling to dereference IDs.
      if (
          data
          and 'inputs' in data.keys()
          and len(data.get('inputs'))
          and 'dataset_name' in kw
      ):
        data['inputs'] = self._reconstitute_inputs(
            data['inputs'], kw['dataset_name']
        )
        # Validate that id and data._id match.
        # TODO(b/171513556): consider removing this if we can simplify the
        # data representation on the frontend so id and meta are not replicated.
        for ex in data['inputs']:
          if ex['id'] != ex['data'].get('_id'):
            raise ValueError(
                'Error: malformed example with inconsistent ids:'
                f' {str(ex)}\nfrom request'
                f' {request.path} {str(request.args.to_dict())}'
            )

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
      metrics: Optional[Mapping[str, lit_components.Metrics]] = None,
      annotators: Optional[list[lit_components.Annotator]] = None,
      layouts: Optional[layout.LitComponentLayouts] = None,
      dataset_loaders: Optional[DatasetLoadersMap] = None,
      model_loaders: Optional[ModelLoadersMap] = None,
      # General server config; see server_flags.py.
      data_dir: Optional[str] = None,
      warm_start: float = 0.0,
      warm_start_progress_indicator: Optional[
          ProgressIndicator
      ] = tqdm.tqdm,  # not in server_flags
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
      enforce_dataset_fields_required: bool = False,
      strict_cache_id_validation: bool = False,
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

    self._caching_model_wrapper_kw = dict(
        cache_dir=self._data_dir,
        strict_id_validation=strict_cache_id_validation,
        id_hash_fn=caching.input_hash,
    )

    # TODO(lit-dev): override layouts instead of merging, to allow clients
    # to opt-out of the default bundled layouts. This will require updating
    # client code to manually merge when this is the desired behavior.
    self._layouts = dict(layout.DEFAULT_LAYOUTS, **(layouts or {}))

    self._model_loaders: ModelLoadersMap = model_loaders or {}
    self._models: dict[str, caching.CachingModelWrapper] = {}
    for name, model in models.items():
      if model_loaders is None:
        # Attempt to infer an init spec for the model before we lose access to
        # the original after wrapping it in a CachingModelWrapper.
        self._model_loaders[name] = (type(model), model.init_spec())
      # Wrap model in caching wrapper and add it to the app
      self._models[name] = caching.CachingModelWrapper(
          model, name, **self._caching_model_wrapper_kw
      )

    self._annotators: list[lit_components.Annotator] = annotators or []
    self._saved_datapoints = {}
    self._saved_datapoints_lock = threading.Lock()

    tmp_datasets: dict[str, lit_dataset.Dataset] = dict(datasets)
    # TODO(b/202210900): get rid of this, just dynamically create the empty
    # dataset on the frontend.
    tmp_datasets[_EMPTY_DATASET_KEY] = lit_dataset.NoneDataset(self._models)

    self._dataset_loaders: DatasetLoadersMap = dataset_loaders or {}
    self._datasets: dict[str, lit_dataset.IndexedDataset] = {}
    for name, ds in tmp_datasets.items():
      if dataset_loaders is None:
        # Attempt to infer an init spec for the dataset before we lose access to
        # the original during dataset annotation and indexing.
        self._dataset_loaders[name] = (type(ds), ds.init_spec())
      # Anotate the dataset
      annotated_ds = self._run_annotators(ds)
      # Index the annotated dataset and add it to the app
      self._datasets[name] = lit_dataset.IndexedDataset(
          base=annotated_ds, id_fn=caching.input_hash)

    # Generator initialization
    if generators is not None:
      self._generators = generators
    else:
      self._generators = core.default_generators()

    # Interpreter initialization
    if interpreters is not None:
      self._interpreters = core.required_interpreters() | interpreters
    else:
      self._interpreters = core.default_interpreters(self._models)

    if metrics is not None:
      self._metrics = metrics
    else:
      self._metrics = core.default_metrics()

    # Component to sync state from TS -> Python. Used in notebooks.
    if sync_state:
      self.ui_state_tracker = ui_state.UIStateTracker()
    else:
      self.ui_state_tracker = None

    # Information on models, datasets, and other components.
    self._info = self._build_metadata()

    # Validate datasets and models if specified.
    self._validate(
        validate,
        enforce_dataset_fields_required=enforce_dataset_fields_required,
        report_all=report_all
    )

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
        '/get_metrics': self._get_metrics,
    }
    wrapped_handlers = {k: self.make_handler(v) for k, v in handlers.items()}

    self._wsgi_app = wsgi_app.App(
        # Wrap endpoint fns to take (handler, request, environ)
        handlers=wrapped_handlers,
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
