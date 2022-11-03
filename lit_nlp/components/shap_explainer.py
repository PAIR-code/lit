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
"""SHAP explanations for datasets and models."""

from typing import Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

import numpy as np
import pandas as pd
import shap

JsonDict = types.JsonDict
Spec = types.Spec

EXPLAIN_KEY = 'Prediction key'
SAMPLE_KEY = 'Sample size'


class TabularShapExplainer(lit_components.Interpreter):
  """SHAP explanations for model predictions over tabular data.

  Kernel SHAP is used to determine the influence that each input features has
  on the prediction of an output feature, given a dataset and model.

  This explainer takes two inputs in its call configuration:

  1.  [Required] The "Prediction key" tells the interpreter which feature in
      the output spec to explain. This interpreter returns None if a
      "Prediction key" is not provided.

  2.  [Optional] "Sample size" is used to extract a random sample from the
      inputs provided to the run() function. If "Sample size" is 0, run() will
      explain the entire inputs list. Due to the performance characteristics of
      SHAP, it is recommended that you use a sample size ≤ 50, otherwise it is
      very likely that the HTTP request will timeout.

  This explainer outputs a list of FeatureSalience objects with the influence of
  each input feature on the predicted value for that input. The size of this
  list is equal to either the length of the inputs or the "Sample size", if the
  latter is > 0. Influence values are normalized in the range of [-1, 1].
  """

  def description(self) -> str:
    return ('Kernel SHAP explanations of input feature influence on model '
            'predictions over tabular data. Influence values are normalized in '
            'the range of [-1, 1].')

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    # Tabular models require all dataset features are present for each datapoint
    compatible_input_types = (types.Scalar, types.CategoryLabel)
    compatible_output_types = (types.MulticlassPreds, types.RegressionScore,
                               types.Scalar, types.SparseMultilabelPreds)
    input_spec_keys = model.input_spec().keys()
    is_tabular = all(
        feature.required and isinstance(feature, compatible_input_types) and
        name in input_spec_keys for name, feature in dataset.spec().items())
    has_outputs = utils.spec_contains(model.output_spec(),
                                      compatible_output_types)
    return is_tabular and has_outputs

  def config_spec(self) -> types.Spec:
    return {
        EXPLAIN_KEY:
            types.SingleFieldMatcher(
                spec='output',
                types=[
                    'MulticlassPreds', 'RegressionScore', 'Scalar',
                    'SparseMultilabelPreds'
                ]),
        SAMPLE_KEY:
            types.Scalar(min_val=0, max_val=50, default=30, step=1),
    }

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.FeatureSalience(autorun=False, signed=True)}

  def run(
      self,
      inputs: list[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None
  ) -> Optional[list[dict[str, dtypes.FeatureSalience]]]:
    """Generate SHAP explanations for model predictions given a set of inputs.

    Args:
      inputs: The subset of inputs from the dataset to get prediction for. If
        empty, falls back to dataset.examples.
      model: The model making the predictions that get explained.
      dataset: The dataset from which the inputs originated.
      model_outputs: Unused, but reqired by the base class.
      config: A dictionary containing the key of the feature to explain, and the
        optional sample size if taking a random sample from the inputs.

    Returns:
      None if the prediction key is not provided or cannot be found in the
      model's output spec.

      A list of FeatureSalience objects, one for each (randomly sampled) input,
      containing per-input feature salience values in the range of [-1, 1].
    """
    config_defaults = {k: v.default for k, v in self.config_spec().items()}
    config = dict(config_defaults, **(config or {}))

    input_feats = list(dataset.spec().keys())
    output_feats = list(model.output_spec().keys())
    pred_key_to_explain = str(config[EXPLAIN_KEY])
    sample_size = int(config[SAMPLE_KEY]) if config[SAMPLE_KEY] else 0

    if not pred_key_to_explain or pred_key_to_explain not in output_feats:
      logging.warning('SHAP requires an output field to explain.')
      return None

    random_baseline = dataset.sample(1).examples
    background = pd.DataFrame(random_baseline)
    inputs_to_use = inputs or dataset.examples
    inputs_to_use = pd.DataFrame(inputs_to_use)

    if sample_size and len(inputs_to_use) > sample_size:
      inputs_to_use = inputs_to_use.sample(sample_size)

    def prediction_fn(examples):
      dict_examples: list[JsonDict] = [{
          input_feats[i]: example[i] for i in range(len(input_feats))
      } for example in examples]

      preds = []

      for pred in model.predict(dict_examples):
        pred_info = model.output_spec()[pred_key_to_explain]

        if isinstance(pred_info, types.MulticlassPreds):
          pred_list: np.ndarray = pred[pred_key_to_explain]
          max_value: float = max(pred_list)
          preds.append(max_value)
        elif isinstance(pred_info, types.SparseMultilabelPreds):
          pred_tuples = pred[pred_key_to_explain]
          pred_list = list(map(lambda pred: pred[1], pred_tuples))  # pytype: disable=annotation-type-mismatch  # enable-nested-classes
          max_value: float = max(pred_list)
          preds.append(max_value)
        else:
          preds.append(pred[pred_key_to_explain])

      return np.array(preds)

    explainer = shap.KernelExplainer(prediction_fn, background)
    values = explainer.shap_values(inputs_to_use)
    salience = [{input_feats[i]: value[i]
                 for i in range(len(input_feats))}
                for value in values]
    salience = [{'saliency': dtypes.FeatureSalience(s)} for s in salience]

    return salience
