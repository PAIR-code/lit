# Copyright 2024 Google LLC
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

"""Constants used across the Model Server and LIT Server code surfaces."""

import enum


class LlmHTTPEndpoints(enum.Enum):
  """Names of HTTP endpoints provided by the Model Server conainer."""

  GENERATE = 'predict'
  SALIENCE = 'salience'
  TOKENIZE = 'tokenize'
