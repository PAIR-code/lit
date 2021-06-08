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
"""AblationFlip generator that ablates input tokens to flip the prediction.

An AblationFlip is defined as a counterfactual input that ablates one or more
tokens in the input at hand in order to obtain a different prediction.

An AblationFlip is considered minimal if no strict subset of the applied token
ablations succeeds in flipping the prediction.

This generator builds on ideas from the following paper.
(1) Local Explanations via Necessity and Sufficiency: Unifying Theory and
    Practice
    David Watson, Limor Gultchin, Ankur Taly, Luciano Floridi
    UAI 2021.
    https://arxiv.org/abs/2103.14651
"""


import copy
import itertools
from typing import Iterator, List, Optional, Tuple

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import cf_utils
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec

PREDICTION_KEY = "Prediction key"
NUM_EXAMPLES_KEY = "Number of examples"
NUM_EXAMPLES_DEFAULT = 5
MAX_ABLATIONS_KEY = "Maximum number of token ablations"
MAX_ABLATIONS_DEFAULT = 3
TOKENS_TO_IGNORE_KEY = "Tokens to freeze"
TOKENS_TO_IGNORE_DEFAULT = []
REGRESSION_THRESH_KEY = "Regression threshold"
REGRESSION_THRESH_DEFAULT = 0.0


class AblationFlip(lit_components.Generator):
  """AblationFlip generator.

  Ablates tokens in input text to generate counterfactuals that flip the
  prediction.

  This generator works for both classification and regression models. In the
  case of classification models, the returned counterfactuals are guaranteed to
  have a different prediction class as the original example. In the case of
  regression models, the returned counterfactuals are guaranteed to be on the
  opposite side of a user-provided threshold as the original example.

  The returned counterfactuals are guaranteed to be minimal in the sense that
  no strict subset of the applied ablations would have resulted in a
  prediction flip.
  """

  def __init__(self):
    # TODO(ataly): Use a more sophisticated tokenizer and detokenizer.
    self.tokenize = str.split
    self.detokenize = " ".join

  def _get_input_text_fields(self,
                             input_spec: JsonDict,
                             example: JsonDict) -> List[str]:
    """Returns names of all text fields in the input_spec."""
    # Find text fields.
    return [key for key in utils.find_spec_keys(input_spec, types.TextSegment)]

  def _subset_exists(self, cand_set, sets):
    """Checks whether a subset of 'cand_set' exists in 'sets'."""
    for s in sets:
      if s.issubset(cand_set):
        return True
    return False

  def _gen_token_idxs_to_ablate(
      self,
      tokens: List[str],
      max_ablations: int,
      tokens_to_ignore: List[str]) -> Iterator[Tuple[int, ...]]:
    """Generates sets of token positions that are eligible for ablation."""
    # Consider all combinations of tokens upto length max_ablations.
    # TODO(ataly): When the total number of tokens is a above a threshold,
    # only consider the top k tokens by attributions for the purpose of
    # ablation.
    token_idxs = np.arange(len(tokens))
    token_idxs_to_ablate = [idx for idx in token_idxs
                            if tokens[idx] not in tokens_to_ignore]
    for i in range(min(len(token_idxs_to_ablate), max_ablations)):
      for s in itertools.combinations(token_idxs_to_ablate, i+1):
        yield s

  def _create_cf(self,
                 example: JsonDict,
                 text_field: str,
                 tokens: List[str],
                 token_idxs: Tuple[int, ...]) -> JsonDict:
    cf = copy.deepcopy(example)
    modified_tokens = [t for i, t in enumerate(tokens)
                       if i not in token_idxs]
    cf[text_field] = self.detokenize(modified_tokens)
    return cf

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_ABLATIONS_KEY: types.TextSegment(
            default=str(MAX_ABLATIONS_DEFAULT)),
        TOKENS_TO_IGNORE_KEY: types.Tokens(default=TOKENS_TO_IGNORE_DEFAULT),
        PREDICTION_KEY: types.FieldMatcher(spec="output",
                                           types=["MulticlassPreds",
                                                  "RegressionScore"]),
        REGRESSION_THRESH_KEY: types.TextSegment(
            default=str(REGRESSION_THRESH_DEFAULT)),
    }

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Identify minimal sets of token albations that alter the prediction."""
    del dataset  # Unused.

    config = config or {}
    num_examples = int(config.get(NUM_EXAMPLES_KEY, NUM_EXAMPLES_DEFAULT))
    max_ablations = int(config.get(MAX_ABLATIONS_KEY, MAX_ABLATIONS_DEFAULT))
    tokens_to_ignore = config.get(TOKENS_TO_IGNORE_KEY,
                                  TOKENS_TO_IGNORE_DEFAULT)
    pred_key = config.get(PREDICTION_KEY, "")
    regression_thresh = float(config.get(REGRESSION_THRESH_KEY,
                                         REGRESSION_THRESH_DEFAULT))
    assert model is not None, "Please provide a model for this generator."

    input_spec = model.input_spec()
    output_spec = model.output_spec()
    assert pred_key, "Please provide the prediction key"
    assert pred_key in output_spec, "Invalid prediction key"

    is_regression = isinstance(output_spec[pred_key], types.RegressionScore)
    if not is_regression:
      assert isinstance(output_spec[pred_key], types.MulticlassPreds), (
          "Only classification or regression models are supported")
    logging.info(r"W3lc0m3 t0 Ablatl0nFl1p \o/")
    logging.info("Original example: %r", example)

    # Get model outputs.
    orig_output = list(model.predict([example]))[0]

    # Get input text segments.
    text_fields = self._get_input_text_fields(input_spec, example)
    assert text_fields, (
        "No text input field found. Cannot generate AblationFlips")

    successful_cfs = []
    for text_field in text_fields:
      if text_field not in example:
        continue
      logging.info("Identifying AblationFlips for input field: %s",
                   str(text_field))
      text = example[text_field]
      tokens = self.tokenize(text)

      max_ablations_for_field = max_ablations
      if input_spec[text_field].required:
        # Update max_ablations_for_field so that it is at most len(tokens) - 1
        # (we don't want to ablate all tokens!).
        max_ablations_for_field = min(len(tokens)-1, max_ablations)
      successful_positions = []
      for token_idxs in self._gen_token_idxs_to_ablate(
          tokens, max_ablations_for_field, tokens_to_ignore):
        if len(successful_cfs) >= num_examples:
          return successful_cfs

        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensures that we only consider
        # sets of tokens that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        # Create counterfactual.
        cf = self._create_cf(example, text_field, tokens, token_idxs)
        # Obtain model prediction.
        cf_output = list(model.predict([cf]))[0]

        if cf_utils.is_prediction_flip(
            cf_output, orig_output, output_spec, pred_key, regression_thresh):
          # Prediction flip found!
          successful_cfs.append(cf)
          successful_positions.append(set(token_idxs))
          if not is_regression:
            # Update label if multi-class prediction.
            cf_utils.update_label(cf, cf_output, output_spec, pred_key)
    return successful_cfs
