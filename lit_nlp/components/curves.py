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
"""An interpreters for generating data for ROC and PR curves."""

from collections.abc import Sequence
from typing import cast, Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils as lit_utils
import numpy as np
from sklearn import metrics

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec

# The config key for specifying model output to use for calculations.
TARGET_PREDICTION_KEY = 'Prediction field'
# The config key for specifying the class label to use for calculations.
TARGET_LABEL_KEY = 'Label'
# They field name in the interpreter output that contains ROC curve data.
ROC_DATA = 'roc_data'
# They field name in the interpreter output that contains PR curve data.
PR_DATA = 'pr_data'


class CurvesInterpreter(lit_components.Interpreter):
  """Returns data for rendering ROC and Precision-Recall curves."""

  def run(self,
          inputs: Sequence[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[Sequence[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    if not config:
      raise ValueError('Curves required config parameters but received none.')

    if (target_label := config.get(TARGET_LABEL_KEY)) is None:
      raise ValueError(
          f'The config \'{TARGET_LABEL_KEY}\' field should contain the positive'
          f' class label.')

    # Find the prediction field key in the model output to use for calculations.
    output_spec = model.output_spec()

    if TARGET_PREDICTION_KEY in config:
      predictions_key: str = config[TARGET_PREDICTION_KEY]
    elif len(pred_keys := self._find_supported_pred_keys(output_spec)) == 1:
      predictions_key: str = pred_keys[0]
    else:
      raise ValueError(
          'Unable to determine prediction field. Please provide one via the'
          f' "{TARGET_PREDICTION_KEY}" field in the CallConfig or update the'
          ' model spec to output a single MulticlassPreds field.'
      )

    if not inputs:
      return {ROC_DATA: [], PR_DATA: []}

    # Run prediction if needed:
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    # Get scores for the target label.
    pred_spec = output_spec.get(predictions_key)
    if not isinstance(pred_spec, types.MulticlassPreds):
      raise TypeError(
          f'Expected {predictions_key} to be a MulticlassPreds field, but got a'
          f' {type(pred_spec).__name__}'
      )

    labels = pred_spec.vocab
    target_index = labels.index(target_label)
    scores = [o[predictions_key][target_index] for o in model_outputs]

    # Get ground truth for the target label.
    parent_key = pred_spec.parent
    ground_truth_list = []
    for ex in inputs:
      ground_truth_label = ex[parent_key]
      ground_truth = 1.0 if ground_truth_label == target_label else 0.0
      ground_truth_list.append(ground_truth)

    # Compute ROC curve data.
    x, y, _ = metrics.roc_curve(ground_truth_list, scores)
    roc_data = list(zip(np.nan_to_num(x), np.nan_to_num(y)))
    roc_data.sort(key=lambda x: x[0])

    # Compute PR curve data.
    x, y, _ = metrics.precision_recall_curve(ground_truth_list, scores)
    pr_data = list(zip(np.nan_to_num(x), np.nan_to_num(y)))
    pr_data.sort(key=lambda x: x[0])

    # Create and return the result.
    return {ROC_DATA: roc_data, PR_DATA: pr_data}

  def is_compatible(
      self, model: lit_model.Model, dataset: lit_dataset.Dataset
  ) -> bool:
    """True if using a classification model and dataset has ground truth."""
    output_spec = model.output_spec()
    supported_keys = self._find_supported_pred_keys(output_spec)
    has_parents = all(
        cast(types.MulticlassPreds, output_spec[key]).parent in dataset.spec()
        for key in supported_keys
    )
    return bool(supported_keys) and has_parents

  def config_spec(self) -> types.Spec:
    # If a model is a multiclass classifier, a user can specify which
    # class label to use for plotting the curves. If the label is not
    # specified then the label with index 0 is used by default.
    return {
        TARGET_LABEL_KEY: types.CategoryLabel(),
        TARGET_PREDICTION_KEY: types.SingleFieldMatcher(
            spec='output', types=['MulticlassPreds'], required=False
        ),
    }

  def meta_spec(self) -> types.Spec:
    return {ROC_DATA: types.CurveDataPoints(), PR_DATA: types.CurveDataPoints()}

  def _find_supported_pred_keys(self, output_spec: types.Spec) -> list[str]:
    """Returns the list of supported prediction keys in the model output.

    Args:
      output_spec: The model output specification.

    Returns:
      The list of keys.
    """
    all_keys = lit_utils.find_spec_keys(output_spec, types.MulticlassPreds)
    supported_keys = [
        k for k in all_keys
        if cast(types.MulticlassPreds, output_spec[k]).parent
    ]
    return supported_keys
