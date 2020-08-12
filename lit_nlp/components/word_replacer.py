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
"""Word replacement generator."""

import copy
import re
from typing import Dict, Tuple, Iterator, List, Text, Optional

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict


class WordReplacer(lit_components.Generator):
  """Word replacement generator."""
  # TODO(lit-team): have multiple replacement options per key
  # TODO(lit-team): automatically handle casing

  # \w+ finds any word that is at least 1 character long. [^\w\s] part finds
  # anything that is not space or word that is exactly 1 character long,
  # splitting all punctuations into their own groups (e.g. hello!!! -> 'hello',
  # '!' , '!', '!').
  tokenization_pattern = re.compile(r'\w+|[^\w\s]')

  def __init__(self, replacements: Optional[Dict[Text, Text]] = None):
    # Populate dictionary with replacement options.
    if replacements is not None:
      assert isinstance(replacements, dict), 'Replacements must be a dict.'
      self.default_replacements = replacements
    else:
      self.default_replacements = {}

  def parse_subs_string(self, subs_string: Text) -> Dict[Text, Text]:
    """Parse a substitutions list of the form 'foo -> bar, spam -> eggs' ."""
    replacements = {}
    rules = subs_string.split(',')
    for rule in rules:
      segments = re.split(r'\s*->\s*', rule)
      if len(segments) != 2:
        logging.warning("Malformed rule: '%s'", rule)
        continue
      src, tgt = segments
      replacements[src.strip()] = tgt.strip()
    return replacements

  def generate_counterfactuals(
      self, text: Text, token_spans: Iterator[Tuple[int, int]],
      replacements: Dict[Text, Text]) -> Iterator[Text]:
    """Replace each token and yield a new string each time that succeeds.

    Args:
      text: input sentence
      token_spans: a list of token position tuples (start, end)
      replacements: a dict of word replacements

    Yields:
      counterfactual: a string
    """
    for start, end in token_spans:
      token = text[start:end]
      if token in replacements:
        yield text[:start] + replacements[token] + text[end:]

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Replace words based on replacement list."""
    del model  # Unused.

    subs_string = config.get('subs') if config else None
    if subs_string:
      replacements = self.parse_subs_string(subs_string)
    else:
      replacements = self.default_replacements

    new_examples = []
    # TODO(lit-dev): move this to generate_all(), so we read the spec once
    # instead of on every example.
    text_keys = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    for text_key in text_keys:
      text_data = example[text_key]
      token_spans = map(lambda x: x.span(),
                        self.tokenization_pattern.finditer(text_data))
      for new_val in self.generate_counterfactuals(text_data, token_spans,
                                                   replacements):
        new_example = copy.deepcopy(example)
        new_example[text_key] = new_val
        new_examples.append(new_example)

    return new_examples
