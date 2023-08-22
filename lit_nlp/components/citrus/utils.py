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
"""Utility functions for explaining text classifiers."""

import numpy as np

DEFAULT_KERNEL_WIDTH = 25


def normalize_scores(scores: np.ndarray,
                     make_positive: bool = False) -> np.ndarray:
  """Makes the absolute values sum to 1, optionally making them all positive."""
  if len(scores) < 1:
    return scores
  scores = scores + np.finfo(np.float32).eps
  if make_positive:
    scores = np.abs(scores)
  return scores / np.abs(scores).sum(-1)


def exponential_kernel(
    distance: float, kernel_width: float = DEFAULT_KERNEL_WIDTH) -> np.ndarray:
  """The exponential kernel."""
  return np.sqrt(np.exp(-(distance**2) / kernel_width**2))
