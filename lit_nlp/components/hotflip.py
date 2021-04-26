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
"""Hotflip generator that perturbs input tokens to flip the prediction.

Hotflip uses a single backward pass to estimate the gradient of each token and
uses them to heuristically estimate the impact of perturbing the token.

This implementation works for both classification and regression models. In the
case of classification models, the returned counterfactuals are guaranteed to
have a different prediction class as the original example. In the case of
regression models, the returned counterfactuals are guaranteed to be on the
opposite side of a user-provided threshold as the original example.
"""

import copy
import itertools
from typing import List, Text, Optional, Type, cast

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

JsonDict = types.JsonDict
Spec = types.Spec

NUM_EXAMPLES_KEY = "Number of examples"
NUM_EXAMPLES_DEFAULT = 5
MAX_FLIPS_KEY = "Maximum number of token flips"
MAX_FLIPS_DEFAULT = 3
TOKENS_TO_IGNORE_KEY = "Tokens to freeze"
TOKENS_TO_IGNORE_DEFAULT = []
DROP_TOKENS_KEY = "Drop tokens instead of flipping"
DROP_TOKENS_DEFAULT = False
REGRESSION_THRESH_KEY = "Flipping threshold for regression score"
REGRESSION_THRESH_DEFAULT = 0.0
MAX_FLIPPABLE_TOKENS = 10


class HotFlip(lit_components.Generator):
  """HotFlip generator.

  A hotflip is defined as a counterfactual sentence that alters one or more
  tokens in the input sentence in order to to obtain a different prediction
  from the input sentence.

  A hotflip is considered minimal if no strict subset of the applied token flips
  succeeds in flipping the prediction.

  This generator is currently only supported on classification models.
  """

  def find_fields(
      self, output_spec: Spec, typ: Type[types.LitType],
      align_field: Optional[Text] = None) -> List[Text]:
    # Find fields of provided 'typ'.
    fields = utils.find_spec_keys(output_spec, typ)

    if align_field is None:
      return fields

    # Only return fields that are aligned to fields with name  specified by
    # 'align_field'.
    ret = []
    for f in fields:
      if "align" in output_spec[f] and output_spec[f].align == align_field:  # pytype: disable=attribute-error,unsupported-operands
        ret.append(f)
    return ret

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_FLIPS_KEY: types.TextSegment(default=str(MAX_FLIPS_DEFAULT)),
        TOKENS_TO_IGNORE_KEY: types.Tokens(default=TOKENS_TO_IGNORE_DEFAULT)
    }

  def _subset_exists(self, cand_set, sets):
    """Checks whether a subset of 'cand_set' exists in 'sets'."""
    for s in sets:
      if s.issubset(cand_set):
        return True
    return False

  def _gen_tokens_to_flip(self, token_idxs, max_flips):
    for i in range(min(len(token_idxs), max_flips)):
      for s in itertools.combinations(token_idxs, i+1):
        yield s

  def _drop_tokens(self, tokens, token_idxs):
    # Returns a copy of 'tokens' with all tokens at indices specified in
    # 'token_idxs' dropped.
    return [t for i, t in enumerate(tokens) if i not in token_idxs]

  def _replace_tokens(self, tokens, token_idxs,
                      replacement_tokens):
    # Returns a copy of 'tokens' with all tokens at indices specified in
    # 'token_idxs' replaced with corresponding tokens in 'replacement_tokens'.
    return [replacement_tokens[j] if j in token_idxs else t
            for j, t in enumerate(tokens)]

  def _get_prediction_key(self, output_spec):
    # Returns prediction key and a boolean indicating whether it corresponds
    # to a regression head.

    # First we look for a classification head.
    pred_keys = self.find_fields(output_spec, types.MulticlassPreds, None)
    if len(pred_keys) == 1:  # pylint: disable=g-explicit-length-test
      return pred_keys[0], False
    # TODO(ataly): Use a config argument when there are multiple prediction
    # heads.

    # If a unique classification head is not found we look for a regression
    # head.
    pred_keys = self.find_fields(output_spec, types.RegressionScore, None)
    if len(pred_keys) == 1:  # pylint: disable=g-explicit-length-test
      return pred_keys[0], True
    return None, False

  def _get_replacement_tokens(self, all_embeddings, vocab,
                              token_grads, direction):
    # Identifies replacement tokens for each token position.
    #
    # 'direction' (which is either -1 or 1) specifies whether the
    # replacement must increase the prediction score (i.e., maximize
    # movement in the direction of the gradients) or decrease prediction
    # score (i.e., maximize movement in opposite direction of the graidents)

    token_grads = token_grads*direction
    # Compute dot product of each (direction adjusted) input token gradient
    # with the embedding table, and pick the argmax.
    # TODO(ataly): Only consider tokens that have the same part-of-speech
    # tag as the original token and/or a certain cosine similarity with the
    # original token.
    replacement_token_ids = np.argmax(
        (np.expand_dims(all_embeddings, 1) @ token_grads.T).squeeze(1), axis=0)
    replacement_tokens = [vocab[id] for id in replacement_token_ids]
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
    drop_tokens = config.get(DROP_TOKENS_KEY, DROP_TOKENS_DEFAULT)
    regression_thresh = float(config.get(REGRESSION_THRESH_KEY,
                                         REGRESSION_THRESH_DEFAULT))

    assert model is not None, "Please provide a model for this generator."
    logging.info(r"W3lc0m3 t0 H0tFl1p \o/")
    logging.info("Original example: %r", example)

    input_spec = model.input_spec()
    output_spec = model.output_spec()
    # Find classification prediciton key.
    pred_key, is_regression = self._get_prediction_key(output_spec)
    if pred_key is None:
      logging.info("Could not find a unique classification or regression head."
                   "Cannot use HotFlip. :-(")
      return []  # Cannot generate examples without gradients.

    # Find gradient fields to use for HotFlip
    grad_fields = self.find_fields(output_spec, types.TokenGradients, None)
    logging.info("Found gradient fields for HotFlip use: %s", str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      logging.info("No gradient fields found. Cannot use HotFlip. :-(")
      return []  # Cannot generate examples without gradients.

    # Get model outputs.
    logging.info("Performing a forward pass on the input example.")
    orig_output = list(model.predict([example]))[0]
    orig_pred = orig_output[pred_key]
    logging.info(orig_output.keys())

    # Get model word embeddings and vocab.
    inv_vocab, embed = model.get_embedding_table()
    assert len(inv_vocab) == embed.shape[0], "Vocab/embeddings size mismatch."
    logging.info("Vocab size: %d, Embedding size: %r", len(inv_vocab),
                 embed.shape)

    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    successful_cfs = []
    successful_positions = []
    for grad_field in grad_fields:
      # Get the tokens and their gradient vectors.
      token_field = output_spec[grad_field].align  # pytype: disable=attribute-error
      tokens = orig_output[token_field]
      grads = orig_output[grad_field]
      token_emb_fields = self.find_fields(output_spec, types.TokenEmbeddings,
                                          token_field)
      assert len(token_emb_fields) == 1, "Found multiple token embeddings"
      token_embs = orig_output[token_emb_fields[0]]
      assert token_embs.shape[0] == grads.shape[0]
      if drop_tokens:
        # Update max_flips so that it is at most len(tokens) - 1 (we don't
        # want to drop all tokens!)
        max_flips = min(len(tokens)-1, max_flips)
      else:
        # Identify replacement tokens.
        direction = -1
        if is_regression:
          # We want the replacements to increase the prediction score if the
          # original score is below the threshold, and decrease otherwise.
          direction = (orig_pred <= regression_thresh)
        replacement_tokens = self._get_replacement_tokens(
            embed, inv_vocab, grads, direction)
      logging.info("Replacement tokens: %s", replacement_tokens)

      # Consider all combinations of tokens upto length max_flips.
      # We will iterate through this list (in toplogically sorted order)
      # and at each iteration, replace the selected tokens with corresponding
      # replacement tokens and checks if the prediction flips.
      # At each level of the topological sort, we will consider combinations
      # by ordering tokens by gradient L2 (i.e., we wish to prioritize flipping
      # tokens that may have the largest impact on the prediction.)
      token_grads_l2 = np.sum(grads * grads, axis=-1)
      # TODO(ataly, bastings): Consider sorting by attributions (either
      # Integrated Gradients or Shapley values).
      token_idxs_sorted_by_grads = np.argsort(token_grads_l2)[::-1]
      token_idxs_to_flip = [idx for idx in token_idxs_sorted_by_grads
                            if tokens[idx] not in tokens_to_ignore]
      # If the number of tokens considered for flipping is larger than
      # MAX_FLIPPABLE_TOKENS we only consider the top tokens.
      token_idxs_to_flip = token_idxs_to_flip[:MAX_FLIPPABLE_TOKENS]

      for token_idxs in self._gen_tokens_to_flip(
          token_idxs_to_flip, max_flips):
        if len(successful_cfs) >= num_examples:
          return successful_cfs
        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensure that we only consider
        # sets of token flips that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        # Create a new counterfactual candidate.
        # TODO(iftenney, bastings): enforce somewhere that this field has the
        # same name in the input and output specs.
        input_token_field = token_field
        input_text_field = input_spec[input_token_field].parent  # pytype: disable=attribute-error
        cf = copy.deepcopy(example)
        if drop_tokens:
          modified_tokens = self._drop_tokens(tokens, token_idxs)
          logging.info("Selected tokens to drop: %s (positions=%s)",
                       [tokens[i] for i in token_idxs], token_idxs)
        else:
          modified_tokens = self._replace_tokens(tokens, token_idxs,
                                                 replacement_tokens)
          logging.info(
              "Selected tokens to flip: %s (positions=%s) with: %s",
              [tokens[i] for i in token_idxs], token_idxs,
              [replacement_tokens[i] for i in token_idxs])
        # TODO(iftenney, bastings): call a model-provided detokenizer here?
        # Though in general tokenization isn't invertible and it's possible for
        # HotFlip to produce wordpiece sequences that don't correspond to any
        # input string.
        cf[input_text_field] = " ".join(modified_tokens)
        cf_output = list(model.predict([cf]))[0]
        cf_pred = cf_output[pred_key]

        if is_regression:
          if (orig_pred <= regression_thresh) != (cf_pred <= regression_thresh):
            # Hotflip found!
            successful_cfs.append(cf)
            successful_positions.append(set(token_idxs))
        else:
          # Classification model.
          if np.argmax(orig_pred) != np.argmax(cf_pred):
            # Hotflip found!
            # TODO(lit-dev): provide a general system for handling labels on
            # generated examples.
            label_key = cast(types.MulticlassPreds,
                             output_spec[pred_key]).parent
            label_names = cast(types.MulticlassPreds,
                               output_spec[pred_key]).vocab
            cf_label = label_names[np.argmax(cf_pred)]
            cf[label_key] = cf_label
            logging.info("Updated counterfactual with new label: %s", cf_label)

            successful_cfs.append(cf)
            successful_positions.append(set(token_idxs))
    return successful_cfs
