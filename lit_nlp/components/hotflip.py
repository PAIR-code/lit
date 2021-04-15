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
"""Hotflip generator that subsitutes an input token with another one.

Hotflip uses a single backward pass to estimate the gradient of substituting a
token with another one, and then substitutes the token with the largest impact
on the loss. This implementation works at the token level.

Paper: https://www.aclweb.org/anthology/P18-2006/
HotFlip: White-Box Adversarial Examples for Text Classification
Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
ACL 2018.
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
      align_typ: Optional[Type[types.LitType]] = None) -> List[Text]:
    # Find fields of provided 'typ'.
    fields = utils.find_spec_keys(output_spec, typ)

    if align_typ is None:
      return fields

    # Check that these are aligned to fields of type 'align_typ'.
    for f in fields:
      align_field = output_spec[f].align  # pytype: disable=attribute-error
      assert align_field in output_spec, "Align field not in output_spec"
      assert isinstance(output_spec[align_field], align_typ)
    return fields

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

    assert model is not None, "Please provide a model for this generator."
    logging.info(r"W3lc0m3 t0 H0tFl1p \o/")
    logging.info("Original example: %r", example)

    # Find classification prediciton key.
    pred_keys = self.find_fields(model.output_spec(),
                                 types.MulticlassPreds, None)
    if len(pred_keys) == 0:  # pylint: disable=g-explicit-length-test
      # TODO(ataly): Add support for regression models.
      logging.warning("The model does not have a classification head."
                      "Cannot use HotFlip. :-(")
      return []  # Cannot generate examples.
    if len(pred_keys) > 1:
      # TODO(ataly): Use a config argument when there are multiple prediction
      # heads.
      logging.warning("Multiple classification heads found."
                      "Cannot use HotFlip. :-(")
      return []  # Cannot generate examples.
    pred_key = pred_keys[0]

    # Find gradient fields to use for HotFlip
    input_spec = model.input_spec()
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec, types.TokenGradients,
                                   types.Tokens)
    logging.info("Found gradient fields for HotFlip use: %s", str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      logging.info("No gradient fields found. Cannot use HotFlip. :-(")
      return []  # Cannot generate examples without gradients.

    # Get model outputs.
    logging.info("Performing a forward/backward pass on the input example.")
    orig_output = list(model.predict([example]))[0]
    logging.info(orig_output.keys())

    # Get model word embeddings and vocab.
    inv_vocab, embed = model.get_embedding_table()
    assert len(inv_vocab) == embed.shape[0], "Vocab/embeddings size mismatch."
    logging.info("Vocab size: %d, Embedding size: %r", len(inv_vocab),
                 embed.shape)

    # Get original prediction class
    orig_probabilities = orig_output[pred_key]
    orig_prediction = np.argmax(orig_probabilities)

    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    successful_counterfactuals = []
    successful_positions = []
    for grad_field in grad_fields:
      # Get the tokens and their gradient vectors.
      token_field = output_spec[grad_field].align  # pytype: disable=attribute-error
      tokens = orig_output[token_field]
      grads = orig_output[grad_field]
      token_emb_fields = self.find_fields(output_spec, types.TokenEmbeddings,
                                          types.Tokens)
      assert len(token_emb_fields) == 1, "Found multiple token embeddings"
      token_embs = orig_output[token_emb_fields[0]]
      assert token_embs.shape[0] == grads.shape[0]

      # We take a dot product of each input token gradient (grads) with the
      # embedding table (embed)
      # TODO(ataly): Only consider tokens that have the same part-of-speech
      # tag as the original token (and a certain cosine similarity with the
      # original token)
      replacement_token_ids = np.argmin(
          (np.expand_dims(embed, 1) @ grads.T).squeeze(1), axis=0)

      replacement_tokens = [inv_vocab[id] for id in replacement_token_ids]
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
        if len(successful_counterfactuals) >= num_examples:
          return successful_counterfactuals
        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensure that we only consider
        # sets of token flips that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        logging.info("Selected tokens to flip: %s (positions=%s) with: %s",
                     [tokens[i] for i in token_idxs], token_idxs,
                     [replacement_tokens[i] for i in token_idxs])

        # Create a new input to the model.
        # TODO(iftenney, bastings): enforce somewhere that this field has the
        # same name in the input and output specs.
        input_token_field = token_field
        input_text_field = input_spec[input_token_field].parent  # pytype: disable=attribute-error
        counterfactual = copy.deepcopy(example)
        modified_tokens = copy.copy(tokens)
        for j in token_idxs:
          modified_tokens[j] = replacement_tokens[j]
        counterfactual[input_token_field] = modified_tokens
        # TODO(iftenney, bastings): call a model-provided detokenizer here?
        # Though in general tokenization isn't invertible and it's possible for
        # HotFlip to produce wordpiece sequences that don't correspond to any
        # input string.
        counterfactual[input_text_field] = " ".join(modified_tokens)

        # Predict a new label for this example.
        counterfactual_output = list(model.predict([counterfactual]))[0]

        # Update label if multi-class prediction.
        # TODO(lit-dev): provide a general system for handling labels on
        # generated examples.
        probabilities = counterfactual_output[pred_key]
        counterfactual_prediction = np.argmax(probabilities)
        label_key = cast(types.MulticlassPreds, output_spec[pred_key]).parent
        label_names = cast(types.MulticlassPreds, output_spec[pred_key]).vocab
        counterfactual_label = label_names[counterfactual_prediction]
        counterfactual[label_key] = counterfactual_label
        logging.info("Updated example with new label: %s", counterfactual_label)

        if counterfactual_prediction != orig_prediction:
          # Hotflip found
          successful_counterfactuals.append(counterfactual)
          successful_positions.append(set(token_idxs))
    return successful_counterfactuals
