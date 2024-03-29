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
"""LIME perturbation-based feature attribution for text sequences."""

from collections.abc import Iterable
import functools
from typing import Any, Optional

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

_SUPPORTED_PRED_TYPES = (types.MulticlassPreds, types.RegressionScore,
                         types.SparseMultilabelPreds)

TARGET_INFO_KEY = '_salience_target'
TARGET_HEAD_KEY = 'Output field to explain'  # TODO(b/205996131): remove
CLASS_KEY = 'Class index to explain'  # TODO(b/205996131): remove
KERNEL_WIDTH_KEY = 'Kernel width'
MASK_KEY = 'Mask'
NUM_SAMPLES_KEY = 'Number of samples'
SEED_KEY = 'Seed'


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
  model_inputs = [
      utils.make_modified_input(original_example, {text_key: s}, 'LIME')
      for s in strings
  ]

  # Get model predictions for the examples.
  model_outputs = model.predict(model_inputs)
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


def get_class_to_explain(
    model: Any,
    pred_key: str,
    example: JsonDict,
    provided_class_to_explain: Optional[int] = None,
) -> Optional[int]:
  """Return the class index to explain.

  The provided class index can be None, in which case this method determines
  which class to explain based on the class with the highest prediction score
  for the provided example.

  Args:
    model: The model to run.
    pred_key: The key to the model's output field to explain.
    example: The example to explain.
    provided_class_to_explain: The class index provided to this explainer, or
      None to use the argmax prediction.

  Returns:
    The true class index to explain, in the range [0, class vocab length).
  """
  pred_spec = model.output_spec()[pred_key]
  # If provided_class_to_explain is -1, then explain the argmax class, for both
  # multiclass and sparse multilabel tasks.
  if provided_class_to_explain is None and isinstance(
      pred_spec, (types.MulticlassPreds, types.SparseMultilabelPreds)
  ):
    pred = list(model.predict([example]))[0][pred_key]
    if isinstance(pred_spec, types.MulticlassPreds):
      return np.argmax(pred)
    else:
      # For sparse multi-label, sort class/score tuples to find the
      # highest-scoring class and get its vocab index.
      pred.sort(key=lambda elem: elem[1], reverse=True)
      class_name_to_explain = pred[0][0]
      return pred_spec.vocab.index(class_name_to_explain)
  else:
    return provided_class_to_explain


class LIME(lit_components.Interpreter):
  """Local Interpretable Model-agnostic Explanations (LIME)."""

  def __init__(
      self,
      autorun: bool = False,
      kernel_width: int = 256,
      mask_token: str = '[MASK]',
      num_samples: int = 256,
      seed: Optional[int] = None,
  ):
    """Creates a LIME interpreter.

    Args:
      autorun: Determines if this intepreter should run automatically.
      kernel_width: Size of the kernel.
      mask_token: Mask token from the tokenizer
      num_samples: Number of samples to take.
      seed: A seed value for the random seed.
    """
    self._autorun: bool = autorun
    self._kernel_width: str = str(kernel_width)
    self._mask_token: str = mask_token
    self._num_samples: str = str(num_samples)
    self._seed: str = str(seed) if seed is not None else ''

  def run(
      self,
      inputs: list[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None,
  ) -> Optional[list[JsonDict]]:
    """Run this component, given a model and input(s)."""
    config_defaults = {k: v.default for k, v in self.config_spec().items()}
    config = dict(config_defaults, **(config or {}))  # update and return

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

    available_pred_keys = utils.find_spec_keys(
        model.output_spec(), _SUPPORTED_PRED_TYPES
    )
    if not available_pred_keys:
      logging.warning('LIME did not find any supported output fields.')
      return None

    if (field := config[TARGET_HEAD_KEY]) and (
        cls_idx := int(config[CLASS_KEY])
    ) != -1:
      # TODO(b/205996131): remove this case
      pred_key = field
      provided_class_to_explain = cls_idx
    elif target_config := config.get(TARGET_INFO_KEY):
      pred_key = target_config['field']
      if pred_key not in available_pred_keys:
        logging.warning("LIME is not compatible with field '%s'", pred_key)
        return None
      # May be None, if there's no label vocab.
      provided_class_to_explain = target_config.get('index')
    else:
      pred_key = available_pred_keys[0]
      provided_class_to_explain = None  # use model prediction

    pred_type_info = model.output_spec()[pred_key]
    all_results = []

    # Explain each input.
    for example in inputs:
      # dict[field name -> interpretations]
      result = {}
      predict_fn = functools.partial(
          _predict_fn,
          model=model,
          original_example=example,
          pred_key=pred_key,
          pred_type_info=pred_type_info,
      )

      class_to_explain = get_class_to_explain(
          model, pred_key, example, provided_class_to_explain
      )

      # Explain each text segment in the input, keeping the others constant.
      for text_key in text_keys:
        input_string = example[text_key]
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
        result[text_key] = dtypes.TokenSalience(explanation.features, scores)

      all_results.append(result)

    return all_results

  def config_spec(self) -> types.Spec:
    return {
        TARGET_INFO_KEY: types.SalienceTargetInfo(),
        # TODO(b/205996131): remove TARGET_HEAD_KEY field
        TARGET_HEAD_KEY: types.SingleFieldMatcher(
            spec='output',
            types=[c.__name__ for c in _SUPPORTED_PRED_TYPES],
            required=False,
        ),
        # TODO(b/205996131): remove CLASS_KEY field
        CLASS_KEY: types.TextSegment(default='-1', required=False),
        MASK_KEY: types.TextSegment(default=self._mask_token, required=False),
        KERNEL_WIDTH_KEY: types.TextSegment(
            default=self._kernel_width, required=False
        ),
        NUM_SAMPLES_KEY: types.TextSegment(
            default=self._num_samples, required=False
        ),
        SEED_KEY: types.TextSegment(default=self._seed, required=False),
    }

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused as salience comes from the model
    text_keys = utils.spec_contains(model.input_spec(), types.TextSegment)
    pred_keys = utils.spec_contains(model.output_spec(), _SUPPORTED_PRED_TYPES)
    return text_keys and pred_keys

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.TokenSalience(autorun=self._autorun, signed=True)}
