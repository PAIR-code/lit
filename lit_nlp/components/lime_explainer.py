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
"""Gradient-based attribution."""

import copy
import functools
from typing import Any, Iterable, List, Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components.citrus import lime
from lit_nlp.components.citrus import utils as citrus_util
from lit_nlp.lib import utils

import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec

CLASS_KEY = 'Class index to explain'
KERNEL_WIDTH_KEY = 'Kernel width'
MASK_KEY = 'Mask'
NUM_SAMPLES_KEY = 'Number of samples'
SEED_KEY = 'Seed'
CLASS_DEFAULT = 1
KERNEL_WIDTH_DEFAULT = 256
MASK_DEFAULT = '[MASK]'
NUM_SAMPLES_DEFAULT = 256
SEED_DEFAULT = None


def new_example(original_example: JsonDict, field: str, new_value: Any):
  """Deep copies the example and replaces `field` with `new_value`."""
  example = copy.deepcopy(original_example)
  example[field] = new_value
  return example


def _predict_fn(strings: Iterable[str], model: Any, original_example: JsonDict,
                text_key: str, pred_key: str):
  """Given raw strings, return probabilities. Used by `lime.explain`."""
  # Prepare example objects to be fed to the model for each sentence/string.
  input_examples = [new_example(original_example, text_key, s) for s in strings]

  # Get model predictions for the examples.
  model_outputs = model.predict(input_examples)
  outputs = np.array([output[pred_key] for output in model_outputs])
  # Make outputs 1D in case of regression or binary classification.
  if outputs.ndim == 2 and outputs.shape[1] == 1:
    outputs = np.squeeze(outputs, axis=1)
  # <float32>[len(strings)] or <float32>[len(strings), num_labels].
  return outputs


class LIME(lit_components.Interpreter):
  """Local Interpretable Model-agnostic Explanations (LIME)."""

  def __init__(self):
    pass

  def run(
      self,
      inputs: List[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[List[JsonDict]] = None,
      config: Optional[JsonDict] = None,
  ) -> Optional[List[JsonDict]]:
    """Run this component, given a model and input(s)."""

    class_to_explain = int(config[CLASS_KEY]) if config else CLASS_DEFAULT
    kernel_width = int(
        config[KERNEL_WIDTH_KEY]) if config else KERNEL_WIDTH_DEFAULT
    mask_string = config[MASK_KEY] if config else MASK_DEFAULT
    num_samples = int(
        config[NUM_SAMPLES_KEY]) if config else NUM_SAMPLES_DEFAULT
    seed = config[SEED_KEY] if config else SEED_DEFAULT

    # Find keys of input (text) segments to explain.
    # Search in the input spec, since it's only useful to look at ones that are
    # used by the model.
    text_keys = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    if not text_keys:
      logging.warning('LIME requires text inputs.')
      return None
    logging.info('Found text fields for LIME attribution: %s', str(text_keys))

    # Find the key of output probabilities field(s).
    pred_keys = utils.find_spec_keys(
        model.output_spec(), (types.MulticlassPreds, types.RegressionScore))
    if not pred_keys:
      logging.warning('LIME did not find any supported output fields.')
      return None

    pred_key = pred_keys[0]  # TODO(lit-dev): configure which prob field to use.
    all_results = []

    # Explain each input.
    for input_ in inputs:
      # Dict[field name -> interpretations]
      result = {}
      predict_fn = functools.partial(
          _predict_fn, model=model, original_example=input_, pred_key=pred_key)

      # Explain each text segment in the input, keeping the others constant.
      for text_key in text_keys:
        input_string = input_[text_key]
        if not input_string:
          logging.info('Could not explain empty string for %s', text_key)
          continue
        logging.info('Explaining: %s', input_string)

        # Perturbs the input string, gets model predictions, fits linear model.
        explanation = lime.explain(
            sentence=input_string,
            predict_fn=functools.partial(predict_fn, text_key=text_key),
            # `class_to_explain` is ignored when predict_fn output is a scalar.
            class_to_explain=class_to_explain,  # Index of the class to explain.
            num_samples=num_samples,
            tokenizer=str.split,
            mask_token=mask_string,
            kernel=functools.partial(
                lime.exponential_kernel, kernel_width=kernel_width),
            seed=seed)

        # Turn the LIME explanation into a list following original word order.
        scores = explanation.feature_importance
        # TODO(lit-dev): Move score normalization to the UI.
        scores = citrus_util.normalize_scores(scores)
        result[text_key] = dtypes.SalienceMap(input_string.split(), scores)

      all_results.append(result)

    return all_results

  def config_spec(self) -> types.Spec:
    return {
        CLASS_KEY: types.TextSegment(default=str(CLASS_DEFAULT)),
        KERNEL_WIDTH_KEY: types.TextSegment(default=str(KERNEL_WIDTH_DEFAULT)),
        MASK_KEY: types.TextSegment(default=MASK_DEFAULT),
        NUM_SAMPLES_KEY: types.TextSegment(default=str(NUM_SAMPLES_DEFAULT)),
        SEED_KEY: types.TextSegment(default=SEED_DEFAULT),
    }

  def is_compatible(self, model: lit_model.Model):
    text_keys = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    pred_keys = utils.find_spec_keys(
        model.output_spec(), (types.MulticlassPreds, types.RegressionScore))
    return len(text_keys) and len(pred_keys)

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.SalienceMap(autorun=False, signed=True)}
