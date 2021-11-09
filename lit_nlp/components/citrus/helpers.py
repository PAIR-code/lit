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
"""Helper classes and functions for explaining text classifiers."""

import math
from typing import Any, List, Optional, Text
import attr
import numpy as np

TOP_K_AVG_RATIO = 0.1

# TODO(xnlp-dev): b/156918552 Maybe merge PosthocExplanation and TextRationale.
# TODO(xnlp-dev): b/156912351 Describe xNLP components in a design doc.


@attr.s(auto_attribs=True)
class PosthocExplanation:
  """Represents a post-hoc explanation with feature importance scores.

  Attributes:
    feature_importance: Feature importance scores for each input feature. These
      are the coefficients of the linear model that was fitted to mimic the
      behavior of a (black-box) prediction function.
    intercept: The intercept of the fitted linear model. This is the independent
      term that is added to make a prediction.
    model: The fitted linear model. An explanation only contains this if it was
      explicitly requested from the explanation method.
    score: The R^2 score of the fitted linear model on the perturbations and
      their labels. This reflects how well the linear model was able to fit to
      the perturbation set.
    prediction: The prediction of the linear model on the full input sentence,
      i.e., an all-true boolean mask.
  """
  feature_importance: np.ndarray
  intercept: Optional[float] = None
  model: Optional[Any] = None
  score: Optional[float] = None
  prediction: Optional[float] = None


class TextRationale:
  """A text with a rationale explanation."""

  def __init__(self,
               text: Text,
               token_weights: List[float],
               top_k_ratio: float = TOP_K_AVG_RATIO):
    """Initializes with a text and a list of token weights.

    Args:
      text: A full-text input to a classifier with tokens separated with ' '.
      token_weights: A list of token weights (in the token position order).
      top_k_ratio: Rationale size in tokens is defined proportional to the input
        length (in tokens). The percentages are given for the ERASER datasets
        and are 10% on average.
    """
    self.text = text
    self.tokens = str.split(text)
    assert len(self.tokens) == len(token_weights), 'Token count does not match.'
    self.token_weights = token_weights

    # Round to the closest equal or larger integer.
    top_k_value = math.ceil(len(self.tokens) * top_k_ratio)

    self.top_k_ids = np.argsort(token_weights)[-top_k_value:]
    self.top_k_ids = list(reversed(self.top_k_ids))
    self.top_k_ids_set = set(self.top_k_ids)

  def get_rationale_text(self, mask_token: Optional[Text] = None) -> str:
    """Returns the text covering only the rationale.

    Args:
      mask_token: Token to use for all the tokens not in the rationale.
    Returns: A string representing the source text with everything but rationale
      masked.
    """
    result = []
    for i, token in enumerate(self.tokens):
      if i in self.top_k_ids_set:
        result.append(token)
      elif mask_token:
        result.append(mask_token)
    return ' '.join(result)

  def get_text_wo_rationale(self, mask_token: Optional[Text] = None) -> str:
    """Returns the text without the rationale.

    Args:
      mask_token: Token to use for all the tokens in the rationale.
    Returns: A string representing the source text with the rationale masked.
    """
    result = []
    for i, token in enumerate(self.tokens):
      if i not in self.top_k_ids_set:
        result.append(token)
      elif mask_token:
        result.append(mask_token)
    return ' '.join(result)
