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
from typing import List, Text, Optional

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
  """HotFlip generator."""

  def find_fields(self, output_spec: Spec) -> List[Text]:
    # Find TokenGradients fields
    grad_fields = utils.find_spec_keys(output_spec, types.TokenGradients)

    # Check that these are aligned to Token fields
    for f in grad_fields:
      tokens_field = output_spec[f].align  # pytype: disable=attribute-error
      assert tokens_field in output_spec, "Tokens field not in output_spec"
      assert isinstance(output_spec[tokens_field], types.Tokens)
    return grad_fields

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None,
               num_examples: int = 1) -> List[JsonDict]:
    """Use gradient to find/substitute the token with largest impact on loss."""
    del dataset  # Unused.

    assert model is not None, "Please provide a model for this generator."
    logging.info(r"W3lc0m3 t0 H0tFl1p \o/")
    logging.info("Original example: %r", example)

    # Find gradient fields to use for HotFlip
    input_spec = model.input_spec()
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info("Found gradient fields for HotFlip use: %s", str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      logging.info("No gradient fields found. Cannot use HotFlip. :-(")
      return []  # Cannot generate examples without gradients.

    # Get model outputs.
    logging.info("Performing a forward/backward pass on the input example.")
    model_output = model.predict_single(example)
    logging.info(model_output.keys())

    # Get model word embeddings and vocab.
    inv_vocab, embed = model.get_embedding_table()
    assert len(inv_vocab) == embed.shape[0], "Vocab/embeddings size mismatch."
    logging.info("Vocab size: %d, Embedding size: %r", len(inv_vocab),
                 embed.shape)

    # Perform a flip in each sequence for which we have gradients (separately).
    # Each sequence may give rise to multiple new examples, depending on how
    # many words we flip.
    # TODO(lit-team): make configurable how many new examples are desired.
    # TODO(lit-team): use only 1 sequence as input (configurable in UI).
    new_examples = []
    for grad_field in grad_fields:

      # Get the tokens and their gradient vectors.
      token_field = output_spec[grad_field].align  # pytype: disable=attribute-error
      tokens = model_output[token_field]
      grads = model_output[grad_field]

      # Identify the token with the largest gradient norm.
      # TODO(lit-team): consider normalizing across all grad fields or just
      # across each one individually.
      grad_norm = np.linalg.norm(grads, axis=1)
      grad_norm = grad_norm / np.sum(grad_norm)  # Match grad attribution value.

      # Get a list of indices of input tokens, sorted by norm, highest first.
      sorted_by_grad_norm = np.argsort(grad_norm)[::-1]

      for i in range(min(num_examples, len(tokens))):
        token_id = sorted_by_grad_norm[i]
        logging.info("Selected token: %s (pos=%d) with gradient norm %f",
                     tokens[token_id], token_id, grad_norm[token_id])
        token_grad = grads[token_id]

        # Take dot product with all word embeddings. Get largest value.
        scores = np.dot(embed, token_grad)

        # TODO(lit-team): Can add criteria to the winner e.g. cosine distance.
        winner = np.argmax(scores)
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
        new_output = model.predict_single(new_example)

        # Update label if multi-class prediction.
        # TODO(lit-dev): provide a general system for handling labels on
        # generated examples.
        for pred_key, pred_type in model.output_spec().items():
          if isinstance(pred_type, types.MulticlassPreds):
            probabilities = new_output[pred_key]
            prediction = np.argmax(probabilities)
            label_key = output_spec[pred_key].parent
            label_names = output_spec[pred_key].vocab
            new_label = label_names[prediction]
            new_example[label_key] = new_label
            logging.info("Updated example with new label: %s", new_label)

        new_examples.append(new_example)

    return new_examples
