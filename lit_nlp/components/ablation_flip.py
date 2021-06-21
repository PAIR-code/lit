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
from typing import Iterator, List, Optional, Text, Tuple

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import cf_utils
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
MAX_ABLATABLE_TOKENS = 10
ABLATED_FIELD_KEY = "ablation_field"


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

  The number of model predictions made by this generator per input field is
  upper-bounded by
  <number of tokens> + 2**MAX_ABLATABLE_TOKENS
  Since MAX_ABLATABLE_TOKENS is a constant, effectively this generator makes
  O(k*n) model predictions, where k is the number of input fields, and n is the
  maximum number of tokens across all fields.
  """

  def __init__(self):
    # TODO(ataly): Use a more sophisticated tokenizer and detokenizer.
    self.tokenize = str.split
    self.detokenize = " ".join

  def _subset_exists(self, cand_set, sets):
    """Checks whether a subset of 'cand_set' exists in 'sets'."""
    for s in sets:
      if s.issubset(cand_set):
        return True
    return False

  def _gen_token_idxs_to_ablate(
      self,
      tokens: List[str],
      loo_scores: List[float],
      max_ablations: int,
      tokens_to_ignore: List[str],
      orig_regression_score: Optional[float] = None,
      regression_thresh: Optional[float] = None) -> Iterator[Tuple[int, ...]]:
    """Generates sets of token positions that are eligible for ablation."""

    # Order tokens by their leave-one-out ablation scores. (Note that these
    # would be negative if the ablation results in reducing the score.) We use
    # ascending order for classification tasks. For regression tasks, the order
    # is based on whether the original score is above or below the threshold --
    # we use ascending order for the former, and descending for the latter.
    token_idxs = np.argsort(loo_scores)
    if regression_thresh and orig_regression_score <= regression_thresh:
      token_idxs = token_idxs[::-1]

    # Drop tokens that the user has asked to ignore.
    token_idxs = [idx for idx in token_idxs
                  if tokens[idx] not in tokens_to_ignore]

    # Only consider the top tokens up to MAX_ABLATABLE_TOKENS.
    token_idxs = token_idxs[:MAX_ABLATABLE_TOKENS]

    # Consider all combinations of tokens upto length max_ablations.
    for i in range(min(len(token_idxs), max_ablations)):
      for s in itertools.combinations(token_idxs, i+1):
        yield s

  def _create_cf(self,
                 example: JsonDict,
                 input_spec: Spec,
                 input_field: str,
                 tokens: List[str],
                 token_idxs: Tuple[int, ...]) -> JsonDict:
    cf = copy.deepcopy(example)
    modified_tokens = [t for i, t in enumerate(tokens)
                       if i not in token_idxs]
    if isinstance(input_spec[input_field], types.TextSegment):
      cf[input_field] = self.detokenize(modified_tokens)
    elif isinstance(input_spec[input_field], types.SparseMultilabel):
      cf[input_field] = modified_tokens
    return cf

  def _generate_leave_one_out_ablation_score(
      self,
      example: JsonDict,
      model: lit_model.Model,
      input_spec: Spec,
      output_spec: Spec,
      orig_output: JsonDict,
      pred_key: Text,
      input_field: str,
      tokens: List[str]) -> List[float]:
    # Returns a list of leave-one-out ablation score for the provided tokens.
    loo_scores = []
    for i in range(len(tokens)):
      cf = self._create_cf(example, input_spec, input_field, tokens,
                           tuple([i]))
      cf_output = list(model.predict([cf]))[0]
      loo_scores.append(cf_utils.prediction_difference(
          cf_output, orig_output, output_spec, pred_key))
    return loo_scores

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_ABLATIONS_KEY: types.TextSegment(
            default=str(MAX_ABLATIONS_DEFAULT)),
        # TODO(ataly,tolgab): Replace this option with one that lets the user
        # freeze entire fields.
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

    successful_cfs = []
    for input_field in input_spec.keys():
      if input_field not in example:
        continue
      if isinstance(input_spec[input_field], types.TextSegment):
        tokens = self.tokenize(example[input_field])
      elif isinstance(input_spec[input_field], types.SparseMultilabel):
        tokens = example[input_field]
      else:
        continue
      logging.info("Identifying AblationFlips for input field: %s",
                   str(input_field))
      max_ablations_for_field = max_ablations
      if input_spec[input_field].required:
        # Update max_ablations_for_field so that it is at most len(tokens) - 1
        # (we don't want to ablate all tokens!).
        max_ablations_for_field = min(len(tokens)-1, max_ablations)

      loo_scores = self._generate_leave_one_out_ablation_score(
          example, model, input_spec, output_spec, orig_output, pred_key,
          input_field, tokens)

      if isinstance(output_spec[pred_key], types.RegressionScore):
        token_idxs_to_ablate = self._gen_token_idxs_to_ablate(
            tokens, loo_scores, max_ablations_for_field, tokens_to_ignore,
            orig_output[pred_key], regression_thresh)
      else:
        # classification
        token_idxs_to_ablate = self._gen_token_idxs_to_ablate(
            tokens, loo_scores, max_ablations_for_field, tokens_to_ignore)

      successful_positions = []
      for token_idxs in token_idxs_to_ablate:
        if len(successful_cfs) >= num_examples:
          return successful_cfs

        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensures that we only consider
        # sets of tokens that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        # Create counterfactual.
        cf = self._create_cf(example, input_spec, input_field,
                             tokens, token_idxs)
        # Obtain model prediction.
        cf_output = list(model.predict([cf]))[0]

        if cf_utils.is_prediction_flip(
            cf_output, orig_output, output_spec, pred_key, regression_thresh):
          # Prediction flip found!
          cf_utils.update_prediction(cf, cf_output, output_spec, pred_key)
          cf[ABLATED_FIELD_KEY] = (
              f"{input_field}{[tokens[idx] for idx in token_idxs]}")
          successful_cfs.append(cf)
          successful_positions.append(set(token_idxs))
    return successful_cfs
