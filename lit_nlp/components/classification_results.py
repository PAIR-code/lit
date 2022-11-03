# Copyright 2022 Google LLC
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
"""An interpreter for analyzing classification results."""

import numbers
from typing import cast, Optional, Sequence

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils as lit_utils
import numpy as np

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


def get_margin_for_input(margin_config: Optional[JsonDict] = None,
                         inp: Optional[JsonDict] = None) -> float:
  """Returns the margin value given a margin config and input example."""
  # When no margin config provided, then the margin of 0 indicates that the
  # class with the highest score is the predicted class.
  if not margin_config:
    return 0

  # Check each facet in the margin config to see if the input matches the
  # facet. If so, then use the margin value for that facet from the config.
  for margin_entry in margin_config.values():
    facet_info = (margin_entry['facetData']['facets']
                  if 'facetData' in margin_entry else {})
    match = True
    if inp:
      for feat, facet_info in facet_info.items():
        value = facet_info['val']
        if (isinstance(inp[feat], numbers.Number) and
            not isinstance(inp[feat], bool)):
          # If the facet is a numeric range string, extract the min and max
          # and check the value against that range.
          min_val = value[0]
          max_val = value[1]
          if not (inp[feat] >= min_val and inp[feat] < max_val):
            match = False
            break
        # If the facet is a standard value, check the feature value for
        # equality to it.
        elif inp[feat] != value:
          match = False
          break
    if match:
      return margin_entry['margin']
  return 0


def get_classifications(
    preds: Sequence[np.ndarray], pred_spec: types.MulticlassPreds,
    margin_config: Optional[Sequence[float]] = None) -> Sequence[int]:
  """Get predicted class indices given prediction scores and configs."""
  # If there is a margin set for the prediction, take the log of the prediction
  # scores and add the margin to the null indexes value before taking argmax
  # to find the predicted class.
  if margin_config is not None:
    null_idx = pred_spec.null_idx
    pred_idxs = []
    null_idx_one_hot = np.eye(len(pred_spec.vocab))[null_idx]
    for p, margin in zip(preds, margin_config):
      logit_mask = margin * null_idx_one_hot
      pred_idx = np.argmax(np.log(p) + logit_mask)
      pred_idxs.append(pred_idx)
  else:
    pred_idxs = [np.argmax(p) for p in preds]
  return pred_idxs


class ClassificationInterpreter(lit_components.Interpreter):
  """Calculates and returns classification results, using thresholds."""

  def run(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      inputs: list[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.IndexedDataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None):

    # Find the prediction field key in the model output to use for calculations.
    output_spec = model.output_spec()
    supported_keys = self._find_supported_pred_keys(output_spec)

    results: list[dict[str, dtypes.ClassificationResult]] = []

    # Run prediction if needed:
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    for i, inp in enumerate(inputs):
      input_result: dict[str, dtypes.ClassificationResult] = {}
      for key in supported_keys:

        margin = get_margin_for_input(
            config[key] if (config and key in config) else None, inp)
        field_spec = cast(types.MulticlassPreds, output_spec[key])
        scores = model_outputs[i][key]
        pred_idx = get_classifications(
            [scores], field_spec, [margin])[0]
        pred_class = field_spec.vocab[pred_idx]
        correct = None
        # If there is ground truth information, calculate error and squared
        # error.
        if (field_spec.parent and field_spec.parent in inp):
          correct = pred_class == inp[field_spec.parent]

        result = dtypes.ClassificationResult(scores, pred_class, correct)
        input_result[key] = result
      results.append(input_result)
    return results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused during model classification
    return lit_utils.spec_contains(model.output_spec(), types.MulticlassPreds)

  def _find_supported_pred_keys(self, output_spec: types.Spec) -> list[str]:
    return lit_utils.find_spec_keys(output_spec, types.MulticlassPreds)
