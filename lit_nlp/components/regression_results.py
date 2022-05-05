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

from typing import Dict, List, Optional, Text

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

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.IndexedDataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):

    # Find the prediction field key in the model output to use for calculations.
    output_spec = model.output_spec()
    supported_keys = self._find_supported_pred_keys(output_spec)

    results: List[Dict[Text, dtypes.RegressionResult]] = []

    # Run prediction if needed:
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    for i, inp in enumerate(inputs):
      input_result: Dict[Text, dtypes.RegressionResult] = {}
      for key in supported_keys:
        if isinstance(model_outputs[i][key], dtypes.RegressionResult):
          continue
        score = model_outputs[i][key]
        error = None
        sq_error = None
        # If there is ground truth information, calculate error and squared
        # error.
        if (output_spec[key].parent and
            output_spec[key].parent in inp):
          ground_truth = inp[output_spec[key].parent]
          error = score - ground_truth
          sq_error = error * error

        result = dtypes.RegressionResult(score, error, sq_error)
        input_result[key] = result
      results.append(input_result)
    return results

  def is_compatible(self, model: lit_model.Model) -> bool:
    output_spec = model.output_spec()
    return True if self._find_supported_pred_keys(output_spec) else False

  def _find_supported_pred_keys(self, output_spec: types.Spec) -> List[Text]:
    return lit_utils.find_spec_keys(output_spec, types.RegressionScore)
