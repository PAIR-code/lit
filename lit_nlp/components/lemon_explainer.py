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
"""Counterfactual explanations using linear model."""

import copy
from typing import Any, List, Optional, Text, Dict, Sequence, Iterable

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components.citrus import lemon
from lit_nlp.components.citrus import utils as citrus_utils
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec


def new_example(original_example: JsonDict, field: Text, new_value: Any):
  """Deep copies the example and replaces `field` with `new_value`."""
  example = copy.deepcopy(original_example)
  example[field] = new_value
  return example


# TODO(lit-dev): Change to calling the CachingModelWrapper for predictions
# instead of using a Dict with the prediction values.
def make_predict_fn(counterfactuals: Dict[str, Sequence[float]]):
  """Makes a predict function that returns pre-computed predictions.

  Since LIT already has cached predictions for the counterfactuals, this mapping
  can be used in place of a function that calls the model.

  Args:
    counterfactuals: a dict mapping counterfactual strings to prediction values.

  Returns:
    A predict function to be used in lemon.explain().
  """
  def _predict_fn(sentences: Iterable[str]):
    return np.array([counterfactuals.get(sentence) for sentence in sentences])
  return _predict_fn


class LEMON(lit_components.Interpreter):
  """LIME-like Explanation Magic Over Novels (LEMON).

  See citrus/lemon.py description for details.
  """

  def __init__(self):
    pass

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
    """Run this component, given a model and input(s)."""
    if not inputs: return

    # Find keys of input (text) segments to explain.
    # Search in the input spec, since it's only useful to look at ones that are
    # used by the model.
    text_keys = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    if not text_keys:
      logging.warning('LEMON requires text inputs.')
      return None
    logging.info('Found text fields for LEMON attribution: %s', str(text_keys))

    pred_key = config['pred_key']
    output_probs = np.array([output[pred_key] for output in model_outputs])

    # Explain the input given counterfactuals.

    # Dict[field name -> interpretations]
    result = {}

    # Explain each text segment in the input, keeping the others constant.
    for text_key in text_keys:
      sentences = [item[text_key] for item in inputs]
      input_to_prediction = dict(zip(sentences, output_probs))

      input_string = sentences[0]
      counterfactuals = sentences[1:]

      # Remove duplicate counterfactuals.
      counterfactuals = list(set(counterfactuals))

      logging.info('Explaining: %s', input_string)

      predict_proba = make_predict_fn(input_to_prediction)

      # Perturbs the input string, gets model predictions, fits linear model.
      explanation = lemon.explain(
          input_string,
          counterfactuals,
          predict_proba,
          class_to_explain=config['class_to_explain'],
          lowercase_tokens=config['lowercase_tokens'])

      scores = np.array(explanation.feature_importance)

      # Normalize feature values.
      scores = citrus_utils.normalize_scores(scores)

      result[text_key] = dtypes.TokenSalience(input_string.split(), scores)

    return [result]
