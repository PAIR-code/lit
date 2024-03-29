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
"""Implementation of PCA as a dimensionality reduction model."""

from absl import logging
from lit_nlp.api import model
from lit_nlp.lib import utils
import numpy as np


class PCAModel(model.ProjectorModel):
  """LIT model API implementation for PCA."""

  def __init__(self, **pca_kw):
    self._fitted = False
    self._num_components = pca_kw["n_components"]

  ##
  # Training methods
  def fit_transform(self, inputs):
    x_input = [i["x"] for i in inputs]
    if not x_input:
      return []
    x_train = np.stack(x_input)
    logging.info("PCA input x_train: %s", str(x_train.shape))

    # Center columns around mean.
    self._mean = np.mean(x_train, 0)
    x_train = x_train - self._mean

    # Find PCA projection.
    cov = np.dot(x_train.T, x_train) / x_train.shape[0]
    evals, evecs = np.linalg.eig(cov)

    # Sort by strongest eigenvalues
    key = np.argsort(evals)[::-1][:self._num_components]
    self._evecs = evecs[:, key]
    self._fitted = True

    # Apply PCA projection
    zs = np.dot(x_train, self._evecs)
    return ({"z": utils.coerce_real(z)} for z in zs)

  ##
  # LIT model API
  def predict_minibatch(self, inputs, **unused_kw):
    if not self._fitted:
      return ({"z": [0, 0, 0]} for _ in inputs)
    x = np.stack([i["x"] for i in inputs])
    x = x - self._mean
    zs = np.dot(x, self._evecs)
    return ({"z": utils.coerce_real(z)} for z in zs)
