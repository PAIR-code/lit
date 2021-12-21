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

import collections
import copy
import itertools
from typing import Iterator, List, Optional, Text, Tuple

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import cf_utils

JsonDict = types.JsonDict
Spec = types.Spec

PREDICTION_KEY = "Prediction key"
NUM_EXAMPLES_KEY = "Number of examples"
NUM_EXAMPLES_DEFAULT = 10
MAX_ABLATIONS_KEY = "Maximum number of token ablations"
MAX_ABLATIONS_DEFAULT = 5
REGRESSION_THRESH_KEY = "Regression threshold"
REGRESSION_THRESH_DEFAULT = 1.0
FIELDS_TO_ABLATE_KEY = "Fields to ablate"
MAX_ABLATABLE_TOKENS = 10
ABLATED_TOKENS_KEY = "ablated_tokens"


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

  The number of model predictions made by this generator is upper-bounded by
  <number of tokens> + 2**MAX_ABLATABLE_TOKENS. Since MAX_ABLATABLE_TOKENS is a
  constant, effectively this generator makes O(n) model predictions, where n is
  the total number of tokens across all input fields.
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

  def _gen_ablation_idxs(
      self,
      loo_scores: List[Tuple[str, int, float]],
      max_ablations: int,
      orig_regression_score: Optional[float] = None,
      regression_thresh: Optional[float] = None
  ) -> Iterator[Tuple[Tuple[str, int], ...]]:
    """Generates sets of token positions that are eligible for ablation."""

    # Order tokens by their leave-one-out ablation scores. (Note that these
    # would be negative if the ablation results in reducing the score.) We use
    # ascending order for classification tasks. For regression tasks, the order
    # is based on whether the original score is above or below the threshold --
    # we use ascending order for the former, and descending for the latter.
    loo_scores = sorted(loo_scores, key=lambda item: item[2])
    ablation_idxs = [(f, idx) for f, idx, _ in loo_scores]
    if regression_thresh and orig_regression_score <= regression_thresh:
      ablation_idxs = ablation_idxs[::-1]

    # Only consider the top tokens up to MAX_ABLATABLE_TOKENS.
    ablation_idxs = ablation_idxs[:MAX_ABLATABLE_TOKENS]

    # Consider all combinations of tokens upto length max_ablations.
    for i in range(min(len(ablation_idxs), max_ablations)):
      for s in itertools.combinations(ablation_idxs, i+1):
        yield s

  def _get_tokens(self,
                  example: JsonDict,
                  input_spec: Spec,
                  input_field: str):
    input_ty = input_spec[input_field]
    if isinstance(input_ty, types.URL):
      return cf_utils.tokenize_url(example[input_field])
    elif isinstance(input_ty, types.SparseMultilabel):
      return example[input_field]
    elif isinstance(input_ty, types.TextSegment):
      return self.tokenize(example[input_field])
    else:
      return []

  def _create_cf(self,
                 example: JsonDict,
                 input_spec: Spec,
                 ablation_idxs: List[Tuple[str, int]]) -> JsonDict:
    # Build a dictionary mapping input fields to the token idxs to be ablated
    # from that field.
    ablation_idxs_per_field = collections.defaultdict(list)
    for field, idx in ablation_idxs:
      ablation_idxs_per_field[field].append(idx)
    ablation_idxs_per_field.default_factory = None  # lock
    cf = copy.deepcopy(example)
    for field, ablation_idxs in ablation_idxs_per_field.items():
      # Original list of tokens at the field.
      orig_tokens = self._get_tokens(example, input_spec, field)

      if (input_spec[field].required
          and len(ablation_idxs) >= len(orig_tokens)):
        # Update token_idxs so that we don't end up ablating all tokens.
        ablation_idxs = ablation_idxs[:-1]

      # Modified list of tokens obtained after ablating the tokens form the
      # indices in ablation_idxs.
      modified_tokens = [t for i, t in enumerate(orig_tokens)
                         if i not in ablation_idxs]

      # Update the field with the modified token list.
      input_ty = input_spec[field]
      if isinstance(input_ty, types.URL):
        url = example[field]
        modified_url = cf_utils.ablate_url_tokens(url, ablation_idxs)
        cf[field] = modified_url
      elif isinstance(input_ty, types.SparseMultilabel):
        cf[field] = modified_tokens
      elif isinstance(input_ty, types.TextSegment):
        cf[field] = self.detokenize(modified_tokens)
    return cf

  def _generate_leave_one_out_ablation_score(
      self,
      example: JsonDict,
      model: lit_model.Model,
      input_spec: Spec,
      output_spec: Spec,
      orig_output: JsonDict,
      pred_key: Text,
      fields_to_ablate: List[str]) -> List[Tuple[str, int, float]]:
    # Returns a list of triples: field, token_idx and leave-one-out score.
    ret = []
    for field in input_spec.keys():
      if field not in example or field not in fields_to_ablate:
        continue
      tokens = self._get_tokens(example, input_spec, field)
      cfs = [self._create_cf(example, input_spec, [(field, i)])
             for i in range(len(tokens))]
      cf_outputs = model.predict(cfs)
      for i, cf_output in enumerate(cf_outputs):
        loo_score = cf_utils.prediction_difference(
            cf_output, orig_output, output_spec, pred_key)
        ret.append((field, i, loo_score))
    return ret

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_ABLATIONS_KEY: types.TextSegment(
            default=str(MAX_ABLATIONS_DEFAULT)),
        PREDICTION_KEY:
            types.FieldMatcher(
                spec="output", types=["MulticlassPreds", "RegressionScore"]),
        REGRESSION_THRESH_KEY:
            types.TextSegment(default=str(REGRESSION_THRESH_DEFAULT)),
        FIELDS_TO_ABLATE_KEY:
            types.MultiFieldMatcher(
                spec="input",
                types=["TextSegment", "SparseMultilabel"],
                select_all=True),
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
    assert model is not None, "Please provide a model for this generator."

    input_spec = model.input_spec()
    pred_key = config.get(PREDICTION_KEY, "")
    regression_thresh = float(config.get(REGRESSION_THRESH_KEY,
                                         REGRESSION_THRESH_DEFAULT))

    output_spec = model.output_spec()
    assert pred_key, "Please provide the prediction key"
    assert pred_key in output_spec, "Invalid prediction key"

    is_regression = isinstance(output_spec[pred_key], types.RegressionScore)
    if not is_regression:
      assert isinstance(output_spec[pred_key], types.MulticlassPreds), (
          "Only classification or regression models are supported")
    logging.info(r"W3lc0m3 t0 Ablatl0nFl1p \o/")
    logging.info("Original example: %r", example)

    # Check for fields to ablate.
    fields_to_ablate = list(config.get(FIELDS_TO_ABLATE_KEY, []))
    if not fields_to_ablate:
      return []

    # Get model outputs.
    orig_output = list(model.predict([example]))[0]
    loo_scores = self._generate_leave_one_out_ablation_score(
        example, model, input_spec, output_spec, orig_output, pred_key,
        fields_to_ablate)

    if isinstance(output_spec[pred_key], types.RegressionScore):
      ablation_idxs_generator = self._gen_ablation_idxs(
          loo_scores, max_ablations,
          orig_output[pred_key], regression_thresh)
    else:
      ablation_idxs_generator = self._gen_ablation_idxs(
          loo_scores, max_ablations)

    tokens_map = {}
    for field in input_spec.keys():
      tokens = self._get_tokens(example, input_spec, field)
      if not tokens:
        continue
      tokens_map[field] = tokens

    successful_cfs = []
    successful_positions = []
    for ablation_idxs in ablation_idxs_generator:
      if len(successful_cfs) >= num_examples:
        return successful_cfs

      # If a subset of the set of tokens have already been successful in
      # obtaining a flip, we continue. This ensures that we only consider
      # sets of tokens that are minimal.
      if self._subset_exists(set(ablation_idxs), successful_positions):
        continue

      # Create counterfactual and obtain model prediction.
      cf = self._create_cf(example, input_spec, ablation_idxs)
      cf_output = list(model.predict([cf]))[0]

      # Check if counterfactual results in a prediction flip.
      if cf_utils.is_prediction_flip(
          cf_output, orig_output, output_spec, pred_key, regression_thresh):
        # Prediction flip found!
        cf_utils.update_prediction(cf, cf_output, output_spec, pred_key)
        cf[ABLATED_TOKENS_KEY] = str(
            [f"{field}[{tokens_map[field][idx]}]"
             for field, idx in ablation_idxs])
        successful_cfs.append(cf)
        successful_positions.append(set(ablation_idxs))
    return successful_cfs
