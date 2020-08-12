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
"""LEMON.

LEMON explains the prediction of a classifier on a particular input using a
list of counterfactuals (i.e., input perturbations) that were derived from that
input. The counterfactuals need to be provided to LEMON together with the input,
and they can originate from e.g., a backtranslation module or a VAE.

It works as follows:

1) Create a vocabulary of unique words in the input sentence. Create a mapping
   from each unique word in the original input to the indices of its
   occurrence.
2) For each counterfactual, create a binary mask indicating for each token
   in the vocabulary if it is present in the counterfactual.
3) Get predictions from the model for the original counterfactuals. Use these as
   labels.
4) Fit a linear model to associate the input word types indicated by the binary
   mask with the resulting predicted label.

The resulting feature importance scores are the linear model coefficients for
the requested output class.
"""
import collections
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from lit_nlp.components.citrus import helpers
import numpy as np
from sklearn import linear_model
from sklearn import metrics

DEFAULT_KERNEL_WIDTH = 25
DEFAULT_MASK_TOKEN = '<unk>'
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_SOLVER = 'lsqr'


def make_vocab_dict(tokens: Sequence[str]) -> Dict[str, Sequence[int]]:
  """Creates a dictionary mapping words in the input sentence to their indices.

  Args:
    tokens: The tokens of the input sentence.

  Returns:
    Ordered vocab dictionary mapping words in the input sentence
    to lists of the indices where they occur.
  """
  vocab_dict = collections.OrderedDict()
  for i, token in enumerate(tokens):
    if token not in vocab_dict:
      vocab_dict[token] = []
    vocab_dict[token].append(i)
  return vocab_dict


def get_masks(counterfactuals: Sequence[Sequence[str]],
              vocab_to_indices: Dict[str, Sequence[int]]):
  """Returns strings with the masked tokens replaced with `mask_token`.

  Args:
    counterfactuals: The tokens of the input's counterfactuals.
    vocab_to_indices: A dictionary mapping unique tokens in the input sentence
        to the indices of its occurrence in the original sentence.

  Returns:
    A list of masks, which are lists of booleans corresponding to tokens.
  """
  masks = []
  for counterfactual in counterfactuals:
    counterfactual_tokens = set(counterfactual)
    masks.append([(token in counterfactual_tokens) for token
                  in vocab_to_indices.keys()])
  return masks


# TODO(ellenj): exponential_kernel() is copied from lime.py - move to util.py.
def exponential_kernel(
    distance: float, kernel_width: float = DEFAULT_KERNEL_WIDTH) -> np.ndarray:
  """The exponential kernel."""
  return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def explain(
    sentence: str,
    counterfactuals: List[str],
    predict_fn: Callable[[Iterable[str]], np.ndarray],
    class_to_explain: int,
    tokenizer: Any = str.split,
    lowercase_tokens: bool = True,
    alpha: float = 1.0,
    solver: str = DEFAULT_SOLVER,
    kernel: Callable[..., np.ndarray] = exponential_kernel,
    distance_fn: Callable[..., np.ndarray] = functools.partial(
        metrics.pairwise.pairwise_distances, metric='cosine'),
    distance_scale: float = 100.,
    return_model: bool = False,
    return_score: bool = False,
    return_prediction: bool = False,
    seed: Optional[int] = None,
) -> helpers.PosthocExplanation:
  """Returns the LEMON explanation for a given sentence.

  By default, this function returns an explanation object containing feature
  importance scores and the intercept. Optionally, more information can be
  returned, such as the linear model, the score of the model on the perturbation
  set, and the prediction that the linear model makes on the original sentence.

  Args:
    sentence: An input to be explained.
    counterfactuals: Counterfactuals generated for this example.
    predict_fn: A prediction function that returns an array of probabilities
      given a list of inputs. The output shape is [len(inputs)] for binary
      classification with scalar output, and [len(inputs), num_classes] for
      multi-class classification.
    class_to_explain: The class ID to explain. E.g., 0 for binary classification
      with scalar output, and 1 for the positive class for two-class
      classification.
    tokenizer: A function that splits the input sentence into tokens.
    lowercase_tokens: Whether to lowercase input and counterfactual tokens.
    alpha: Regularization strength of the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    solver: Solver to use in the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    kernel: A kernel function to be used on the distance function. By default,
      the exponential kernel (with kernel width DEFAULT_KERNEL_WIDTH) is used.
    distance_fn: A distance function to use in range [0, 1]. Default: cosine.
    distance_scale: A scalar factor multiplied with the distances before the
      kernel is applied.
    return_model: Returns the fitted linear model.
    return_score: Returns the score of the linear model on the perturbations.
      This is the R^2 of the linear model predictions w.r.t. their targets.
    return_prediction: Returns the prediction of the linear model on the full
      original sentence.
    seed: Optional random seed to make the explanation deterministic, if the
      solver is specified to be stochastic (i.e. 'sag' or 'saga').

  Returns:
    The explanation for the requested class.
  """
  # Include the original sentence in the list of counterfactuals for
  # explanation.
  counterfactuals.append(sentence)

  tokens = tokenizer(sentence)
  counterfactual_tokens = [tokenizer(sentence) for sentence in counterfactuals]

  if lowercase_tokens:
    tokens = [token.lower() for token in tokens]

    for i in range(len(counterfactual_tokens)):
      counterfactual_tokens[i] = [token.lower()
                                  for token in counterfactual_tokens[i]]

  vocab_to_indices = make_vocab_dict(tokens)

  masks = get_masks(
      counterfactual_tokens, vocab_to_indices)
  all_true_mask = np.ones_like(masks[0], dtype=np.bool)

  probs = predict_fn(counterfactuals)
  probs = probs[:, class_to_explain]  # We are only interested in 1 class.

  distances = distance_fn(all_true_mask.reshape(1, -1), masks).flatten()
  distances = distance_scale * distances
  distances = kernel(distances)

  # Fit a linear model for the requested output class.
  model = linear_model.Ridge(
      alpha=alpha, solver=solver, random_state=seed).fit(
          masks, probs, sample_weight=distances)

  # Convert the original sentence tokens to their respective importance values.
  vocab_importance = model.coef_
  feature_importance = np.zeros_like(tokens, dtype=np.float32)
  for token, importance in zip(vocab_to_indices.keys(), vocab_importance):
    for token_index in vocab_to_indices[token]:
      feature_importance[token_index] = importance

  # Add option to return coefficients
  explanation = helpers.PosthocExplanation(
      feature_importance=feature_importance, intercept=model.intercept_)

  if return_model:
    explanation.model = model

  if return_score:
    explanation.score = model.score(masks, probs)

  if return_prediction:
    explanation.prediction = model.predict(all_true_mask.reshape(1, -1))

  return explanation
