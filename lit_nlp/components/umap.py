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
"""Implementation of UMAP as a dimensionality reduction model."""

from absl import logging
from lit_nlp.components import projection
import numpy as np
import umap


class UmapModel(projection.ProjectorModel):
  """LIT model API implementation for UMAP."""

  def __init__(self, **umap_kw):
    self._umap = umap.UMAP(**umap_kw)
    self._fitted = False

  ##
  # Training methods
  def fit_transform(self, inputs):
    x_input = [i["x"] for i in inputs]
    if not x_input:
      return []
    x_train = np.stack(x_input)
    logging.info("UMAP input x_train: %s", str(x_train.shape))
    zs = self._umap.fit_transform(x_train)
    self._fitted = True
    return ({"z": z} for z in zs)

  ##
  # LIT model API
  def predict_minibatch(self, inputs, **unused_kw):
    if not self._fitted:
      return ({"z": [0, 0, 0]} for i in inputs)
    x = np.stack([i["x"] for i in inputs])
    zs = self._umap.transform(x)
    return ({"z": z} for z in zs)
