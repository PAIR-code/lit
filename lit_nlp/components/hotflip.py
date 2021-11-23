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
"""HotFlip generator that perturbs input tokens to flip the prediction.

A HotFlip is defined as a counterfactual input that alters one or more
tokens in the input at hand in order to obtain a different prediction.

A hotflip is considered minimal if no strict subset of the applied token flips
succeeds in flipping the prediction.

This generator extends ideas from the following papers.
(1) HotFlip: White-Box Adversarial Examples for Text Classification
    Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
    ACL 2018.
    https://www.aclweb.org/anthology/P18-2006/

(2) Local Explanations via Necessity and Sufficiency: Unifying Theory and
    Practice
    David Watson, Limor Gultchin, Ankur Taly, Luciano Floridi
    UAI 2021.
    https://arxiv.org/abs/2103.14651
"""

import copy
import itertools
from typing import Iterator, List, Optional, Text, Tuple, Type

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
MAX_FLIPS_KEY = "Maximum number of token flips"
MAX_FLIPS_DEFAULT = 3
TOKENS_TO_IGNORE_KEY = "Tokens to freeze"
TOKENS_TO_IGNORE_DEFAULT = []
REGRESSION_THRESH_KEY = "Regression threshold"
REGRESSION_THRESH_DEFAULT = 0.0
MAX_FLIPPABLE_TOKENS = 10
FIELDS_TO_HOTFLIP_KEY = "Fields to hotflip"


class HotFlip(lit_components.Generator):
  """HotFlip generator.

  This implemention uses a single backward pass to estimate the gradient of
  each token and uses them to heuristically estimate the impact of perturbing
  the token.

  This generator works for both classification and regression models. In the
  case of classification models, the returned counterfactuals are guaranteed to
  have a different prediction class as the original example. In the case of
  regression models, the returned counterfactuals are guaranteed to be on the
  opposite side of a user-provided threshold as the original example.

  The returned counterfactuals are guaranteed to be minimal in the sense that
  no strict subset of the applied token perturbations would have resulted in a
  prediction flip.
  """

  def find_fields(
      self, spec: Spec, typ: Type[types.LitType],
      align_field: Optional[Text] = None) -> List[Text]:
    # Find fields of provided 'typ'.
    fields = utils.find_spec_keys(spec, typ)

    if align_field is None:
      return fields

    # Only return fields that are aligned to fields with name specified by
    # align_field.
    return [f for f in fields
            if getattr(spec[f], "align", None) == align_field]

  def _get_tokens_and_gradients(self,
                                input_spec: JsonDict,
                                output_spec: JsonDict,
                                output: JsonDict,
                                selected_fields: List[str]):
    """Returns a dictionary mapping token fields to tokens and gradients."""
    # Find selected token fields.
    input_spec_keys = set(utils.find_spec_keys(input_spec, types.Tokens))
    logging.info("input_spec_keys: %r", input_spec_keys)
    selected_input_spec_keys = list(input_spec_keys & set(selected_fields))
    logging.info("selected_input_spec_keys: %r", selected_input_spec_keys)
    token_fields = [key for key in selected_input_spec_keys
                    if input_spec[key].is_compatible(output_spec.get(key))]

    if len(token_fields) == 0:  # pylint: disable=g-explicit-length-test
      return {}

    ret = {}
    for token_field in token_fields:
      # Get tokens, token gradients and token embeddings.
      tokens = output[token_field]
      grad_fields = self.find_fields(output_spec, types.TokenGradients,
                                     token_field)
      assert grad_fields, (
          f"No gradients found for {token_field}. Cannot use HotFlip. :-(")
      assert len(grad_fields) == 1, (
          f"Multiple gradients found for {token_field}."
          f"Cannot use HotFlip. :-(")
      grads = output[grad_fields[0]] if grad_fields else None
      ret[token_field] = [tokens, grads]
    return ret

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_FLIPS_KEY: types.TextSegment(default=str(MAX_FLIPS_DEFAULT)),
        TOKENS_TO_IGNORE_KEY: types.Tokens(default=TOKENS_TO_IGNORE_DEFAULT),
        PREDICTION_KEY: types.FieldMatcher(spec="output",
                                           types=["MulticlassPreds",
                                                  "RegressionScore"]),
        REGRESSION_THRESH_KEY: types.TextSegment(
            default=str(REGRESSION_THRESH_DEFAULT)),
        FIELDS_TO_HOTFLIP_KEY:
            types.MultiFieldMatcher(
                spec="input",
                types=["Tokens"],
                select_all=True),
    }

  def _subset_exists(self, cand_set, sets):
    """Checks whether a subset of 'cand_set' exists in 'sets'."""
    for s in sets:
      if s.issubset(cand_set):
        return True
    return False

  def _gen_token_idxs_to_flip(
      self,
      tokens: List[str],
      token_grads: np.ndarray,
      max_flips: int,
      tokens_to_ignore: List[str]) -> Iterator[Tuple[int, ...]]:
    """Generates sets of token positions that are eligible for flipping."""
    # Consider all combinations of tokens upto length max_flips.
    # We will iterate through this list (sortted by cardinality) and at each
    # iteration, replace the selected tokens with corresponding replacement
    # tokens and checks if the prediction flips. At each cardinality, we will
    # consider combinations by ordering tokens by gradient L2 in order to
    # prioritize flipping tokens that may have the largest impact on the
    # prediction.
    token_idxs = np.arange(len(tokens))
    token_grads_l2 = np.sum(token_grads * token_grads, axis=-1)
    # TODO(ataly, bastings): Consider sorting by attributions (either
    # Integrated Gradients or Shapley values).
    token_idxs = np.argsort(token_grads_l2)[::-1]
    token_idxs_to_flip = [idx for idx in token_idxs
                          if tokens[idx] not in tokens_to_ignore]
    # If the number of tokens considered for flipping is larger than
    # MAX_FLIPPABLE_TOKENS we only consider the top tokens.
    token_idxs_to_flip = token_idxs_to_flip[:MAX_FLIPPABLE_TOKENS]

    for i in range(min(len(token_idxs_to_flip), max_flips)):
      for s in itertools.combinations(token_idxs_to_flip, i+1):
        yield s

  def _flip_tokens(self,
                   tokens: List[str],
                   token_idxs: Tuple[int, ...],
                   replacement_tokens: List[str]) -> List[str]:
    """Perturbs tokens at the indices specified in 'token_idxs'."""
    modified_tokens = [replacement_tokens[j] if j in token_idxs else t
                       for j, t in enumerate(tokens)]
    return modified_tokens

  def _create_cf(self,
                 example: JsonDict,
                 token_field: str,
                 text_field: str,
                 tokens: List[str],
                 token_idxs: Tuple[int, ...],
                 replacement_tokens: List[str]) -> JsonDict:
    cf = copy.deepcopy(example)
    modified_tokens = self._flip_tokens(
        tokens, token_idxs, replacement_tokens)
    # TODO(iftenney, bastings): call a model-provided detokenizer here?
    # Though in general tokenization isn't invertible and it's possible for
    # HotFlip to produce wordpiece sequences that don't correspond to any
    # input string.
    cf[token_field] = modified_tokens
    cf[text_field] = " ".join(modified_tokens)
    return cf

  def _get_replacement_tokens(
      self,
      embedding_matrix: np.ndarray,
      inv_vocab: List[Text],
      token_grads: np.ndarray,
      orig_output: JsonDict,
      direction: int = -1) -> List[str]:
    """Identifies replacement tokens for each token position."""
    token_grads = token_grads * direction
    # Compute dot product of each input token gradient with the embedding
    # matrix, and pick the argmin.
    # TODO(ataly): Only consider tokens that have the same part-of-speech
    # tag as the original token and/or a certain cosine similarity with the
    # original token.
    replacement_token_ids = np.argmax(
        (np.expand_dims(embedding_matrix, 1) @ token_grads.T).squeeze(1),
        axis=0)
    replacement_tokens = [inv_vocab[id] for id in replacement_token_ids]
    return replacement_tokens

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Identify minimal sets of token flips that alter the prediction."""
    del dataset  # Unused.

    config = config or {}
    num_examples = int(config.get(NUM_EXAMPLES_KEY, NUM_EXAMPLES_DEFAULT))
    max_flips = int(config.get(MAX_FLIPS_KEY, MAX_FLIPS_DEFAULT))
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

    is_regression = False
    if isinstance(output_spec[pred_key], types.RegressionScore):
      is_regression = True
    else:
      assert isinstance(output_spec[pred_key], types.MulticlassPreds), (
          "Only classification or regression models are supported")
    logging.info(r"W3lc0m3 t0 H0tFl1p \o/")
    logging.info("Original example: %r", example)

    # Get model outputs.
    orig_output = list(model.predict([example]))[0]

    # Check config for selected fields.
    selected_fields = list(config.get(FIELDS_TO_HOTFLIP_KEY, []))
    if not selected_fields:
      return []

    # Get tokens (corresponding to each text input field) and corresponding
    # gradients.
    tokens_and_gradients = self._get_tokens_and_gradients(
        input_spec, output_spec, orig_output, selected_fields)
    assert tokens_and_gradients, (
        "No token fields found. Cannot use HotFlip. :-(")

    # Copy tokens into input example.
    example = copy.deepcopy(example)
    for token_field, v in tokens_and_gradients.items():
      tokens, _ = v
      example[token_field] = tokens

    inv_vocab, embedding_matrix = model.get_embedding_table()
    assert len(inv_vocab) == embedding_matrix.shape[0], (
        "Vocab/embeddings size mismatch.")

    successful_cfs = []
    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    # TODO(lit-team): Refactor the following code so that it's not so deeply
    # nested (and easier to track loop state).
    for token_field, v in tokens_and_gradients.items():
      tokens, grads = v
      text_field = input_spec[token_field].parent  # pytype: disable=attribute-error
      logging.info("Identifying Hotflips for input field: %s", str(text_field))
      direction = -1
      if is_regression:
        # We want the replacements to increase the prediction score if the
        # original score is below the threshold, and decrease otherwise.
        direction = (1 if orig_output[pred_key] <= regression_thresh else -1)
      replacement_tokens = self._get_replacement_tokens(
          embedding_matrix, inv_vocab, grads, direction)

      successful_positions = []
      for token_idxs in self._gen_token_idxs_to_flip(
          tokens, grads, max_flips, tokens_to_ignore):
        if len(successful_cfs) >= num_examples:
          return successful_cfs
        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensures that we only consider
        # sets of token flips that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        # Create counterfactual.
        cf = self._create_cf(example, token_field, text_field, tokens,
                             token_idxs, replacement_tokens)
        # Obtain model prediction.
        cf_output = list(model.predict([cf]))[0]

        if cf_utils.is_prediction_flip(
            cf_output, orig_output, output_spec, pred_key, regression_thresh):
          # Prediciton flip found!
          cf_utils.update_prediction(cf, cf_output, output_spec, pred_key)
          successful_cfs.append(cf)
          successful_positions.append(set(token_idxs))
    return successful_cfs
