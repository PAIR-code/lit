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

A hotflip is defined as a counterfactual sentence that alters one or more
tokens in the input sentence in order to to obtain a different prediction
from the input sentence.

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
    arxiv 2021
    https://arxiv.org/abs/2103.14651
"""

import copy
import itertools
from typing import Iterator, List, Optional, Text, Tuple, Type, cast

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
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
DROP_TOKENS_KEY = "Drop tokens instead of flipping"
DROP_TOKENS_DEFAULT = False
MAX_FLIPPABLE_TOKENS = 10


class HotFlip(lit_components.Generator):
  """HotFlip generator.

  This implemention uses a single backward pass to estimate the gradient of
  each token and uses them to heuristically estimate the impact of perturbing
  the token.

  This generator is currently only supported on classification models.
  """

  def find_fields(
      self, output_spec: Spec, typ: Type[types.LitType],
      align_field: Optional[Text] = None) -> List[Text]:
    # Find fields of provided 'typ'.
    fields = utils.find_spec_keys(output_spec, typ)

    if align_field is None:
      return fields

    # Only return fields that are aligned to fields with name specified by
    # align_field.
    return [f for f in fields
            if getattr(output_spec[f], "align", None) == align_field]

  def config_spec(self) -> types.Spec:
    return {
        NUM_EXAMPLES_KEY: types.TextSegment(default=str(NUM_EXAMPLES_DEFAULT)),
        MAX_FLIPS_KEY: types.TextSegment(default=str(MAX_FLIPS_DEFAULT)),
        TOKENS_TO_IGNORE_KEY: types.Tokens(default=TOKENS_TO_IGNORE_DEFAULT),
        DROP_TOKENS_KEY: types.TextSegment(default=str(DROP_TOKENS_DEFAULT)),
        PREDICTION_KEY: types.FieldMatcher(spec="output",
                                           types=["MulticlassPreds"])
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
      tokens_to_ignore: List[str],
      drop_tokens: Optional[bool] = False) -> Iterator[Tuple[int, ...]]:
    """Generates sets of token positions that are eligible for flipping."""
    if drop_tokens:
      # Update max_flips so that it is at most len(tokens) - 1 (we don't
      # want to drop all tokens!)
      max_flips = min(len(tokens)-1, max_flips)

    # Consider all combinations of tokens upto length max_flips.
    # We will iterate through this list (sortted by cardinality) and at each
    # iteration, replace the selected tokens with corresponding replacement
    # tokens and checks if the prediction flips. At each cardinality, we will
    # consider combinations by ordering tokens by gradient L2 in order to
    # prioritize flipping tokens that may have the largest impact on the
    # prediction.
    token_grads_l2 = np.sum(token_grads * token_grads, axis=-1)
    # TODO(ataly, bastings): Consider sorting by attributions (either
    # Integrated Gradients or Shapley values).
    token_idxs_sorted_by_grads = np.argsort(token_grads_l2)[::-1]
    token_idxs_to_flip = [idx for idx in token_idxs_sorted_by_grads
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
                   drop_tokens: bool,
                   replacement_tokens: List[str]) -> List[str]:
    """Perturbs tokens at the indices specified in 'token_idxs'."""
    # TODO(iftenney, bastings): enforce somewhere that this field has the
    # same name in the input and output specs.
    if drop_tokens:
      modified_tokens = [t for i, t in enumerate(tokens)
                         if i not in token_idxs]
      logging.info("Selected tokens to drop: %s (positions=%s)",
                   [tokens[i] for i in token_idxs], token_idxs)
    else:
      modified_tokens = [replacement_tokens[j] if j in token_idxs else t
                         for j, t in enumerate(tokens)]
      logging.info(
          "Selected tokens to flip: %s (positions=%s) with: %s",
          [tokens[i] for i in token_idxs], token_idxs,
          [replacement_tokens[i] for i in token_idxs])
    return modified_tokens

  def _update_label(self,
                    example: JsonDict,
                    example_output: JsonDict,
                    output_spec: Spec,
                    pred_key: Text):
    """Updates prediction label in the provided example assuming a classification model."""
    probabilities = example_output[pred_key]
    pred_class = np.argmax(probabilities)
    label_key = cast(types.MulticlassPreds, output_spec[pred_key]).parent
    label_names = cast(types.MulticlassPreds, output_spec[pred_key]).vocab
    example_label = label_names[pred_class]
    example[label_key] = example_label

  def _is_hotflip(self,
                  cf_output: JsonDict,
                  orig_output: JsonDict,
                  pred_key: Text) -> bool:
    """Check if cf_output and orig_output specify different prediction classes."""
    cf_pred_class = np.argmax(cf_output[pred_key])
    orig_pred_class = np.argmax(orig_output[pred_key])
    return cf_pred_class != orig_pred_class

  def _get_replacement_tokens(
      self,
      embedding_matrix: np.ndarray,
      inv_vocab: List[Text],
      token_grads: np.ndarray) -> List[str]:
    """Identifies replacement tokens for each token position."""
    # Compute dot product of each input token gradient with the embedding
    # matrix, and pick the argmin.
    # TODO(ataly): Only consider tokens that have the same part-of-speech
    # tag as the original token and/or a certain cosine similarity with the
    # original token.
    replacement_token_ids = np.argmin(
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
    drop_tokens = bool(config.get(DROP_TOKENS_KEY, DROP_TOKENS_DEFAULT))
    pred_key = config.get(PREDICTION_KEY, "")
    assert model is not None, "Please provide a model for this generator."

    input_spec = model.input_spec()
    output_spec = model.output_spec()
    assert pred_key, "Please provide the prediction key"
    assert pred_key in output_spec, "Invalid prediction key"
    assert isinstance(output_spec[pred_key], types.MulticlassPreds), (
        "Only classification models are supported")
    logging.info(r"W3lc0m3 t0 H0tFl1p \o/")
    logging.info("Original example: %r", example)

    # Find gradient fields to use for HotFlip
    grad_fields = self.find_fields(output_spec, types.TokenGradients,
                                   None)
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      logging.info("No gradient fields found. Cannot use HotFlip. :-(")
      return []  # Cannot generate examples without gradients.
    logging.info("Found gradient fields for HotFlip use: %s", str(grad_fields))

    # Get model outputs.
    logging.info("Performing a forward/backward pass on the input example.")
    orig_output = list(model.predict([example]))[0]
    logging.info(orig_output.keys())

    # Get model word embeddings and vocab.
    inv_vocab, embedding_matrix = model.get_embedding_table()
    assert len(inv_vocab) == embedding_matrix.shape[0], (
        "Vocab/embeddings size mismatch.")
    logging.info("Vocab size: %d, Embedding size: %r", len(inv_vocab),
                 embedding_matrix.shape)

    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    successful_cfs = []
    successful_positions = []
    # TODO(lit-team): Refactor the following code so that it's not so deeply
    # nested (and easier to track loop state).
    for grad_field in grad_fields:
      # Get tokens, token gradients and token embeddings.
      token_field = output_spec[grad_field].align  # pytype: disable=attribute-error
      tokens = orig_output[token_field]
      grads = orig_output[grad_field]
      token_emb_fields = self.find_fields(output_spec, types.TokenEmbeddings,
                                          token_field)
      assert len(token_emb_fields) == 1, "Found multiple token embeddings"
      token_embs = orig_output[token_emb_fields[0]]
      assert token_embs.shape[0] == grads.shape[0]

      replacement_tokens = None
      if not drop_tokens:
        replacement_tokens = self._get_replacement_tokens(
            embedding_matrix, inv_vocab, grads)
        logging.info("Replacement tokens: %s", replacement_tokens)

      for token_idxs in self._gen_token_idxs_to_flip(
          tokens, grads, max_flips, tokens_to_ignore, drop_tokens):
        if len(successful_cfs) >= num_examples:
          return successful_cfs
        # If a subset of the set of tokens have already been successful in
        # obtaining a flip, we continue. This ensures that we only consider
        # sets of token flips that are minimal.
        if self._subset_exists(set(token_idxs), successful_positions):
          continue

        # Create counterfactual.
        cf = copy.deepcopy(example)
        modified_tokens = self._flip_tokens(
            tokens, token_idxs, drop_tokens, replacement_tokens)
        cf[token_field] = modified_tokens
        # TODO(iftenney, bastings): call a model-provided detokenizer here?
        # Though in general tokenization isn't invertible and it's possible for
        # HotFlip to produce wordpiece sequences that don't correspond to any
        # input string.
        text_field = input_spec[token_field].parent  # pytype: disable=attribute-error
        cf[text_field] = " ".join(modified_tokens)

        # Get model outputs.
        cf_output = list(model.predict([cf]))[0]

        if self._is_hotflip(cf_output, orig_output, pred_key):
          # Hotflip found
          # Update label if multi-class prediction.
          # TODO(lit-dev): provide a general system for handling labels on
          # generated examples.
          self._update_label(cf, cf_output, output_spec, pred_key)

          successful_cfs.append(cf)
          successful_positions.append(set(token_idxs))
    return successful_cfs

