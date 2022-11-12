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
"""An interpreter for analyzing regression results."""

from typing import cast, Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils as lit_utils

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec


class RegressionInterpreter(lit_components.Interpreter):
  """Calculates and returns regression results from model outputs."""

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

    results: list[dict[str, dtypes.RegressionResult]] = []

    # Run prediction if needed:
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    for i, inp in enumerate(inputs):
      input_result: dict[str, dtypes.RegressionResult] = {}
      for key in supported_keys:
        field_spec = cast(types.RegressionScore, output_spec[key])
        score = model_outputs[i][key]
        error = None
        sq_error = None
        # If there is ground truth information, calculate error and squared
        # error.
        if (field_spec.parent and field_spec.parent in inp):
          ground_truth = inp[field_spec.parent]
          error = score - ground_truth
          sq_error = error * error

        result = dtypes.RegressionResult(score, error, sq_error)
        input_result[key] = result
      results.append(input_result)
    return results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused as regressions depend on model only
    return lit_utils.spec_contains(model.output_spec(), types.RegressionScore)

  def _find_supported_pred_keys(self, output_spec: types.Spec) -> list[str]:
    return lit_utils.find_spec_keys(output_spec, types.RegressionScore)
