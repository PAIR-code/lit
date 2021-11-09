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
"""Simple scrambling test generator."""

import copy
import random
from typing import List, Text, Optional

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict


class Scrambler(lit_components.Generator):
  """Scramble all words in an example to generate a new example."""

  @staticmethod
  def scramble(val: Text) -> Text:
    words = val.split(' ')
    random.shuffle(words)
    return ' '.join(words)

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Naively scramble all words in an example."""
    del model  # Unused.
    del config  # Unused.

    # TODO(lit-dev): move this to generate_all(), so we read the spec once
    # instead of on every example.
    text_keys = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    if not text_keys:
      return []
    new_example = copy.deepcopy(example)
    for text_key in text_keys:
      new_example[text_key] = self.scramble(example[text_key])
    return [new_example]
