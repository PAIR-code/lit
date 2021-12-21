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

TARGET_HEAD_KEY = 'Output field to explain'
CLASS_KEY = 'Class index to explain'
KERNEL_WIDTH_KEY = 'Kernel width'
MASK_KEY = 'Mask'
NUM_SAMPLES_KEY = 'Number of samples'
SEED_KEY = 'Seed'


def new_example(original_example: JsonDict, field: str, new_value: Any):
  """Deep copies the example and replaces `field` with `new_value`."""
  example = copy.deepcopy(original_example)
  example[field] = new_value
  return example


def _predict_fn(strings: Iterable[str], model: Any, original_example: JsonDict,
                text_key: str, pred_key: str, pred_type_info: types.LitType):
  """Given raw strings, return scores. Used by `lime.explain`.

  Adjust the `original_example` by changing the value of the field `text_key`
  by the values in `strings`, and run model prediction on each adjusted example,
  returning a list of scores, one entry per adjusted example.

  Args:
    strings: The adjusted strings to set in the original example.
    model: The model to run.
    original_example: The original example to adjust.
    text_key: The field in which to adjust the original example with the
      provided strings.
    pred_key: The key to the model's output field to explain.
    pred_type_info: The `LitType` value for the model's output field to explain.

  Returns:
    A list of scores for the model output on the adjusted examples. For
    regression tasks, a 1D list of values. For classification and multi-label,
    a 2D list, where each entry is a list of scores for each possible label,
    in order by the class index.
  """
  # Prepare example objects to be fed to the model for each sentence/string.
  input_examples = [new_example(original_example, text_key, s) for s in strings]

  # Get model predictions for the examples.
  model_outputs = model.predict(input_examples)
  outputs = [output[pred_key] for output in model_outputs]
  if isinstance(pred_type_info, types.SparseMultilabelPreds):
    assert pred_type_info.vocab, (
        f'No vocab found for {pred_key} field. Cannot use LIME.')
    # Convert list of class/score tuples to a list of scores for each possible
    # class, in class index order.
    output_arr = np.zeros([len(outputs), len(pred_type_info.vocab)])
    for i, pred in enumerate(outputs):
      for p in pred:
        class_idx = pred_type_info.vocab.index(p[0])
        output_arr[i, class_idx] = p[1]

    outputs = output_arr
  else:
    # Make outputs 1D in case of regression or binary classification.
    # <float32>[len(strings)] or <float32>[len(strings), num_labels].
    outputs = np.array(outputs)
    if outputs.ndim == 2 and outputs.shape[1] == 1:
      outputs = np.squeeze(outputs, axis=1)
  return outputs


def get_class_to_explain(provided_class_to_explain: int, model: Any,
                         pred_key: str, example: JsonDict) -> int:
  """Return the class index to explain.

  The provided class index can be -1, in which case this method determines
  which class to explain based on the class with the highest prediction score
  for the provided example.

  Args:
    provided_class_to_explain: The class index provided to this explainer.
    model: The model to run.
    pred_key: The key to the model's output field to explain.
    example: The example to explain.

  Returns:
    The true class index to explain, in the range [0, class vocab length).
  """
  pred_type_info = model.output_spec()[pred_key]
  # If provided_class_to_explain is -1, then explain the argmax class, for both
  # multiclass and sparse multilabel tasks.
  if ((isinstance(pred_type_info, types.MulticlassPreds) or
       isinstance(pred_type_info, types.SparseMultilabelPreds)) and
      provided_class_to_explain == -1):
    pred = list(model.predict([example]))[0][pred_key]
    if isinstance(pred_type_info, types.MulticlassPreds):
      return np.argmax(pred)
    else:
      # For sparse multi-label, sort class/score tuples to find the
      # highest-scoring class and get its vocab index.
      pred.sort(key=lambda elem: elem[1], reverse=True)
      class_name_to_explain = pred[0][0]
      return pred_type_info.vocab.index(class_name_to_explain)
  else:
    return provided_class_to_explain


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
    config_defaults = {k: v.default for k, v in self.config_spec().items()}
    config = dict(config_defaults, **(config or {}))  # update and return

    provided_class_to_explain = int(config[CLASS_KEY])
    kernel_width = int(config[KERNEL_WIDTH_KEY])
    num_samples = int(config[NUM_SAMPLES_KEY])
    mask_string = (config[MASK_KEY])
    # pylint: disable=g-explicit-bool-comparison
    seed = int(config[SEED_KEY]) if config[SEED_KEY] != '' else None
    # pylint: enable=g-explicit-bool-comparison

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
        model.output_spec(),
        (types.MulticlassPreds, types.RegressionScore,
         types.SparseMultilabelPreds))
    if not pred_keys:
      logging.warning('LIME did not find any supported output fields.')
      return None
    pred_key = config[TARGET_HEAD_KEY] or pred_keys[0]

    all_results = []

    # Explain each input.
    for input_ in inputs:
      # Dict[field name -> interpretations]
      result = {}
      predict_fn = functools.partial(
          _predict_fn, model=model, original_example=input_, pred_key=pred_key,
          pred_type_info=model.output_spec()[pred_key])

      class_to_explain = get_class_to_explain(
          provided_class_to_explain, model, pred_key, input_)

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
            class_to_explain=class_to_explain,
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
        result[text_key] = dtypes.TokenSalience(input_string.split(), scores)

      all_results.append(result)

    return all_results

  def config_spec(self) -> types.Spec:
    matcher_types = ['MulticlassPreds', 'SparseMultilabelPreds',
                     'RegressionScore']
    return {
        TARGET_HEAD_KEY: types.FieldMatcher(spec='output', types=matcher_types),
        CLASS_KEY: types.TextSegment(default='-1'),
        MASK_KEY: types.TextSegment(default='[MASK]'),
        KERNEL_WIDTH_KEY: types.TextSegment(default='256'),
        NUM_SAMPLES_KEY: types.TextSegment(default='256'),
        SEED_KEY: types.TextSegment(default=''),
    }

  def is_compatible(self, model: lit_model.Model):
    text_keys = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    pred_keys = utils.find_spec_keys(
        model.output_spec(),
        (types.MulticlassPreds, types.RegressionScore,
         types.SparseMultilabelPreds))
    return len(text_keys) and len(pred_keys)

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.TokenSalience(autorun=False, signed=True)}
