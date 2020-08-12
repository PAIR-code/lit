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

from typing import cast, List, Text, Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec


class GradientNorm(lit_components.Interpreter):
  """Salience map from gradient L2 norm."""

  def find_fields(self, output_spec: Spec) -> List[Text]:
    # Find TokenGradients fields
    grad_fields = utils.find_spec_keys(output_spec, types.TokenGradients)

    # Check that these are aligned to Tokens fields
    for f in grad_fields:
      tokens_field = output_spec[f].align  # pytype: disable=attribute-error
      assert tokens_field in output_spec
      assert isinstance(output_spec[tokens_field], types.Tokens)
    return grad_fields

  def _interpret(self, grads: np.ndarray, tokens: np.ndarray):
    assert grads.shape[0] == len(tokens)
    # Norm of dy/d(embs)
    grad_norm = np.linalg.norm(grads, axis=1)
    grad_norm /= np.sum(grad_norm)
    # <float32>[num_tokens]
    return grad_norm

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Run this component, given a model and input(s)."""
    # Find gradient fields to interpret
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info('Found fields for gradient attribution: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    all_results = []
    for o in model_outputs:
      # Dict[field name -> interpretations]
      result = {}
      for grad_field in grad_fields:
        token_field = cast(types.TokenGradients, output_spec[grad_field]).align
        tokens = o[token_field]
        scores = self._interpret(o[grad_field], tokens)
        result[grad_field] = dtypes.SalienceMap(tokens, scores)
      all_results.append(result)

    return all_results
