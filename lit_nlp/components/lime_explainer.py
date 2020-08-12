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
from typing import cast, Any, List, Text, Optional

from absl import logging
from lime import lime_text
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec


def new_example(original_example: JsonDict, field: Text, new_value: Any):
  """Deep copies the example and replaces `field` with `new_value`."""
  example = copy.deepcopy(original_example)
  example[field] = new_value
  return example


def explanation_to_array(explanation: Any):
  """Given a LIME explanation object, return a numpy array with scores."""
  # local_exp is a List[(word_position, score)]. We need to sort it.
  scores = sorted(explanation.local_exp[1])  # Puts it back in word order.
  scores = np.array([v for k, v in scores])
  scores = scores / np.abs(scores).sum()
  return scores


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
      kernel_width: int = 25,  # TODO(lit-dev): make configurable in UI.
      mask_string: str = '[MASK]',  # TODO(lit-dev): make configurable in UI.
      num_samples: int = 256,  # TODO(lit-dev): make configurable in UI.
  ) -> Optional[List[JsonDict]]:
    """Run this component, given a model and input(s)."""

    # Find keys of input (text) segments to explain.
    # Search in the input spec, since it's only useful to look at ones that are
    # used by the model.
    text_keys = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    if not text_keys:
      logging.warning('LIME requires text inputs.')
      return None
    logging.info('Found text fields for LIME attribution: %s', str(text_keys))

    # Find the key of output probabilities field(s).
    pred_keys = utils.find_spec_keys(model.output_spec(), types.MulticlassPreds)
    if not pred_keys:
      logging.warning('LIME did not find a multi-class predictions field.')
      return None

    pred_key = pred_keys[0]  # TODO(lit-dev): configure which prob field to use.
    pred_spec = cast(types.MulticlassPreds, model.output_spec()[pred_key])
    label_names = pred_spec.vocab

    # Create a LIME text explainer instance.
    explainer = lime_text.LimeTextExplainer(
        class_names=label_names,
        split_expression=str.split,
        kernel_width=kernel_width,
        mask_string=mask_string,  # This is the string used to mask words.
        bow=False)  # bow=False masks inputs, instead of deleting them entirely.

    all_results = []

    # Explain each input.
    for input_ in inputs:
      # Dict[field name -> interpretations]
      result = {}

      # Explain each text segment in the input, keeping the others constant.
      for text_key in text_keys:
        input_string = input_[text_key]
        logging.info('Explaining: %s', input_string)

        # Use the number of words as the number of features.
        num_features = len(input_string.split())

        def _predict_proba(strings: List[Text]):
          """Given raw strings, return probabilities. Used by `explainer`."""
          input_examples = [new_example(input_, text_key, s) for s in strings]
          model_outputs = model.predict(input_examples)
          probs = np.array([output[pred_key] for output in model_outputs])
          return probs  # <float32>[len(strings), num_labels]

        # Perturbs the input string, gets model predictions, fits linear model.
        explanation = explainer.explain_instance(
            input_string,
            _predict_proba,
            num_features=num_features,
            num_samples=num_samples)

        # Turn the LIME explanation into a list following original word order.
        scores = explanation_to_array(explanation)
        result[text_key] = dtypes.SalienceMap(input_string.split(), scores)

      all_results.append(result)

    return all_results
