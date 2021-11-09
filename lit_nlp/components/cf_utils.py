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
"""Utility functions for generating counterfactuals."""

import re
from typing import List, Optional, Text, Tuple, cast

from lit_nlp.api import types
import numpy as np


def update_prediction(example: types.JsonDict,
                      example_output: types.JsonDict,
                      output_spec: types.JsonDict,
                      pred_key: Text):
  """Updates prediction score and label (if classification model) in the provided example."""
  prediction = example_output[pred_key]
  example[pred_key] = prediction
  pred_spec = output_spec[pred_key]
  if isinstance(pred_spec, types.MulticlassPreds):
    # Update label
    # TODO(lit-dev): provide a general system for handling labels on
    # generated examples.
    pred_spec = cast(types.MulticlassPreds, pred_spec)
    label_key = pred_spec.parent
    label_names = pred_spec.vocab
    pred_class = np.argmax(prediction)
    example_label = label_names[pred_class]
    example[label_key] = example_label


def is_prediction_flip(cf_output: types.JsonDict,
                       orig_output: types.JsonDict,
                       output_spec: types.JsonDict,
                       pred_key: Text,
                       regression_thresh: Optional[float] = None) -> bool:
  """Check if cf_output and  orig_output specify different prediciton classes."""
  if isinstance(output_spec[pred_key], types.RegressionScore):
    # regression model. We use the provided threshold to binarize the output.
    cf_pred_class = (cf_output[pred_key] <= regression_thresh)
    orig_pred_class = (orig_output[pred_key] <= regression_thresh)
  else:
    cf_pred_class = np.argmax(cf_output[pred_key])
    orig_pred_class = np.argmax(orig_output[pred_key])
  return cf_pred_class != orig_pred_class


def prediction_difference(cf_output: types.JsonDict,
                          orig_output: types.JsonDict,
                          output_spec: types.JsonDict,
                          pred_key: Text) -> float:
  """Returns the difference in prediction between cf_output and orig_output."""
  if isinstance(output_spec[pred_key], types.RegressionScore):
    # regression model. We use the provided threshold to binarize the output.
    cf_pred = cf_output[pred_key]
    orig_pred = orig_output[pred_key]
  else:
    orig_pred_class = np.argmax(orig_output[pred_key])
    cf_pred = cf_output[pred_key][orig_pred_class]
    orig_pred = orig_output[pred_key][orig_pred_class]
  return cf_pred - orig_pred


def _tokenize_url(url: str) -> List[Tuple[str, int, int]]:
  """Tokenizes a URL and returns list of triples specifying the token string and its start and end position."""
  if not url:
    return []
  separator_regex = "[^a-zA-Z0-9]"
  separator_matches = list(re.finditer(separator_regex, url))
  if not separator_matches:
    # If no separator is found, use the entire string.
    return [(url, 0, len(url))]

  tokens = []  # a list of tuples of token, start index, end index.
  start_idx = 0  # start index for next token
  for i in range(len(separator_matches)):
    sep = separator_matches[i]
    sep_start_idx = sep.start()
    if sep_start_idx > start_idx:
      tokens.append((url[start_idx:sep_start_idx], start_idx, sep_start_idx))
    start_idx = sep.end()

  if start_idx < len(url):
    tokens.append((url[start_idx:], start_idx, len(url)))
  return tokens


def tokenize_url(url: str) -> List[str]:
  """Tokenizes the provided URL and returns a list of token strings."""
  url_tokens = _tokenize_url(url)
  return [t for t, _, _ in url_tokens]


def ablate_url_tokens(url: str,
                      token_idxs_to_ablate: Tuple[int, ...]) -> str:
  """Ablates the tokens at the provided indices and returns the resulting URL."""
  url_tokens = _tokenize_url(url)
  start = 0
  modified_url_pieces = []
  token_idxs_to_ablate = sorted(token_idxs_to_ablate)
  for token_idx in token_idxs_to_ablate:
    assert token_idx < len(url_tokens), (
        "token_idxs_to_ablate must all fall in the range 0 to number of tokens"
        " returned by tokenize_url")
    _, token_start, token_end = url_tokens[token_idx]
    modified_url_pieces.append(url[start:token_start])
    start = token_end
  modified_url_pieces.append(url[start:])
  return "".join(modified_url_pieces)
