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

from typing import Optional, Text, cast

from lit_nlp.api import types
import numpy as np


def update_label(example: types.JsonDict,
                 example_output: types.JsonDict,
                 output_spec: types.JsonDict,
                 pred_key: Text):
  """Updates prediction label in the provided example assuming a classification model."""
  # TODO(lit-dev): provide a general system for handling labels on
  # generated examples.
  assert isinstance(output_spec[pred_key], types.MulticlassPreds)
  probabilities = example_output[pred_key]
  pred_class = np.argmax(probabilities)
  label_key = cast(types.MulticlassPreds, output_spec[pred_key]).parent
  label_names = cast(types.MulticlassPreds, output_spec[pred_key]).vocab
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
