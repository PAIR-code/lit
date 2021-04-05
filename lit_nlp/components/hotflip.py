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


class HotFlip(lit_components.Generator):
  """HotFlip generator.

  A hotflip is defined as a counterfactual sentence that alters one or more
  tokens in the input sentence in order to to obtain a different prediction
  from the input sentence.

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

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None,
               num_examples: int = 1) -> List[JsonDict]:
    """Use gradient to find/substitute the token with largest impact on loss."""
    # TODO(lit-team): This function is quite long. Consider breaking it
    # into small functions.
    del dataset  # Unused.

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

    # Perform a flip in each sequence for which we have gradients (separately).
    # Each sequence may give rise to multiple new examples, depending on how
    # many words we flip.
    # TODO(lit-team): make configurable how many new examples are desired.
    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    new_examples = []
    for grad_field in grad_fields:
      # Get the tokens and their gradient vectors.
      token_field = output_spec[grad_field].align  # pytype: disable=attribute-error
      tokens = orig_output[token_field]
      grads = orig_output[grad_field]
      token_emb_fields = self.find_fields(output_spec, types.TokenEmbeddings,
                                          types.Tokens)
      assert len(token_emb_fields) == 1, "Found multiple token embeddings"
      token_embs = orig_output[token_emb_fields[0]]

      # Identify the token with the largest gradient attribution,
      # defined as the dot product between the token embedding and gradient
      # of the output wrt the embedding.
      assert token_embs.shape[0] == grads.shape[0]
      token_grad_attrs = np.sum(token_embs * grads, axis=-1)
      # Get a list of indices of input tokens, sorted by gradient attribution,
      # highest first. We will flip tokens in this order.
      sorted_by_grad_attrs = np.argsort(token_grad_attrs)[::-1]

      for i in range(min(num_examples, len(tokens))):
        token_id = sorted_by_grad_attrs[i]
        logging.info("Selected token: %s (pos=%d) with gradient attribution %f",
                     tokens[token_id], token_id, token_grad_attrs[token_id])
        token_grad = grads[token_id]

        # Take dot product with all word embeddings. Get smallest value.
        # (We are look for a replacement token that will lower the score
        # the current class, thereby increasing the chances of a label
        # flip.)
        # TODO(lit-team): Can add criteria to the winner e.g. cosine distance.
        scores = np.dot(embed, token_grad)
        winner = np.argmin(scores)
        logging.info("Replacing [%s] (pos=%d) with option %d: [%s] (id=%d)",
                     tokens[token_id], token_id, i, inv_vocab[winner], winner)

        # Create a new input to the model.
        # TODO(iftenney, bastings): enforce somewhere that this field has the
        # same name in the input and output specs.
        input_token_field = token_field
        input_text_field = input_spec[input_token_field].parent  # pytype: disable=attribute-error
        new_example = copy.deepcopy(example)
        modified_tokens = copy.copy(tokens)
        modified_tokens[token_id] = inv_vocab[winner]
        new_example[input_token_field] = modified_tokens
        # TODO(iftenney, bastings): call a model-provided detokenizer here?
        # Though in general tokenization isn't invertible and it's possible for
        # HotFlip to produce wordpiece sequences that don't correspond to any
        # input string.
        new_example[input_text_field] = " ".join(modified_tokens)

        # Predict a new label for this example.
        new_output = list(model.predict([new_example]))[0]

        # Update label if multi-class prediction.
        # TODO(lit-dev): provide a general system for handling labels on
        # generated examples.
        probabilities = new_output[pred_key]
        new_prediction = np.argmax(probabilities)
        label_key = cast(types.MulticlassPreds, output_spec[pred_key]).parent
        label_names = cast(types.MulticlassPreds, output_spec[pred_key]).vocab
        new_label = label_names[new_prediction]
        new_example[label_key] = new_label
        logging.info("Updated example with new label: %s", new_label)

        if new_prediction != orig_prediction:
          # Hotflip found
          new_examples.append(new_example)
        else:
          # We make new_example as our base example and continue with more
          # token flips.
          example = new_example
          tokens = modified_tokens
    return new_examples
