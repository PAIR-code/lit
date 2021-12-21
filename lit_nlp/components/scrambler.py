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

FIELDS_TO_SCRAMBLE_KEY = 'Fields to scramble'


class Scrambler(lit_components.Generator):
  """Scramble all words in an example to generate a new example."""

  @staticmethod
  def scramble(val: Text) -> Text:
    words = val.split(' ')
    random.shuffle(words)
    return ' '.join(words)

  def config_spec(self) -> types.Spec:
    return {
        FIELDS_TO_SCRAMBLE_KEY:
            types.MultiFieldMatcher(
                spec='input',
                types=['TextSegment'],
                select_all=True),
    }

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Naively scramble all words in an example.

    Note: Even if more than one field is to be scrambled, only a single example
    will be produced, unlike other generators which will produce multiple
    examples, one per field.

    Args:
      example: the example used for basis of generated examples.
      model: the model.
      dataset: the dataset.
      config: user-provided config properties.

    Returns:
      examples: a list of generated examples.
    """
    del model  # Unused.

    config = config or {}

    # If config key is missing, generate no examples.
    fields_to_scramble = list(config.get(FIELDS_TO_SCRAMBLE_KEY, []))
    if not fields_to_scramble:
      return []

    # TODO(lit-dev): move this to generate_all(), so we read the spec once
    # instead of on every example.
    text_keys = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    if not text_keys:
      return []

    text_keys = [key for key in text_keys if key in fields_to_scramble]

    new_example = copy.deepcopy(example)
    for text_key in text_keys:
      new_example[text_key] = self.scramble(example[text_key])
    return [new_example]
