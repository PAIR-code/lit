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
from typing import Dict, Iterator, List, Text, Optional, Pattern

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict

IGNORE_CASING_KEY = 'ignore_casing'
SUBSTITUTIONS_KEY = 'Substitutions'
FIELDS_TO_REPLACE_KEY = 'Fields to replace'


class WordReplacer(lit_components.Generator):
  """Generate new examples by replacing words in examples.

  Substitutions must be of the form 'foo -> bar, spam -> eggs'.
  """

  def __init__(self, replacements: Optional[Dict[Text, List[Text]]] = None):
    # Populate dictionary with replacement options.
    if replacements is not None:
      assert isinstance(replacements, dict), 'Replacements must be a dict.'
      assert all([isinstance(tgt, list) for tgt in replacements.values()
                 ]), 'Replacement dict must be Text->List[Text]'
      self.default_replacements = replacements
    else:
      self.default_replacements = {}

  def parse_subs_string(self, subs_string: Text,
                        ignore_casing: bool = True) -> Dict[Text, List[Text]]:
    """Parse a substitutions list of the form 'foo -> bar, spam -> eggs' ."""
    replacements = {}
    # TODO(lit-dev) Use pyparsing if the pattern gets more complicated.
    rules = subs_string.split(',')
    for rule in rules:
      segments = re.split(r'\s*->\s*', rule)
      if len(segments) != 2:
        logging.warning("Malformed rule: '%s'", rule)
        continue
      src, tgt_str = segments
      tgts = re.split(r'\s*\|\s*', tgt_str)
      if ignore_casing:
        src = src.lower()
      replacements[src.strip()] = [tgt.strip() for tgt in tgts]
    return replacements

  def _get_replacement_pattern(self,
                               replacements: Dict[Text, List[Text]],
                               ignore_casing: bool = True) -> Pattern[str]:
    r"""Generate replacement pattern for whole word match.

    If the source word does not end or begin with non-word characters
    (e.g. punctuation) then do a whole word match by using the special word
    boundary character "\b". This allows us to replace whole words ignoring
    punctuation around them (e.g. "cat." becomes "dog." with rule "cat"->"dog").
    However, if the source word has a non-word character at its edges this
    fails. For example for the rule "."-> "," it would not find "cat. " as there
    is no boundary between "." and " ". Therefore, for patterns with punctuation
    at the word boundaries, we ignore the whole word match and replace all
    instances. So "cat.dog" will become "dogdog" for "cat."->"dog" instead of
    being ignored. Also "." -> "," will replace all instances of "." with ",".

    Args:
      replacements: A dict of word replacements
      ignore_casing: Ignore casing for source words if True.

    Returns:
      regexp_pattern: Compiled regexp pattern used to find source words in
                      replacements.
    """
    re_strings = []
    for s in replacements:
      pattern_str = r'\b%s\b' % re.escape(s)
      # If the source word ends or begins with a non-word character (see above.)
      if not re.search(pattern_str, s):
        pattern_str = r'%s' % re.escape(s)
      re_strings.append(pattern_str)

    casing_flag = re.IGNORECASE if ignore_casing else 0

    return re.compile('|'.join(re_strings), casing_flag)

  def generate_counterfactuals(
      self, text: Text,
      replacement_regex: Pattern[str],
      replacements: Dict[Text, List[Text]],
      ignore_casing: bool = True) -> Iterator[Text]:
    """Replace each token and yield a new string each time that succeeds.

    Note: ignores casing.

    Args:
      text: input sentence
      replacement_regex: The regexp string used to find source words.
      replacements: Dictionary of source and replacement tokens.
      ignore_casing: Ignore casing if this is true.

    Yields:
      counterfactual: a string
    """
    # If replacement_regex is empty do not attempt to match.
    if not replacement_regex.pattern:
      return

    for s in replacement_regex.finditer(text):
      start, end = s.span()
      token = text[start:end]
      if ignore_casing:
        token = token.lower()
      # Yield one output for each target.
      for tgt in replacements[token]:
        yield text[:start] + tgt + text[end:]

  def generate(self,
               example: JsonDict,
               model: lit_model.Model,
               dataset: lit_dataset.Dataset,
               config: Optional[JsonDict] = None) -> List[JsonDict]:
    """Replace words based on replacement list.

    Note: If multiple fields are selected for replacement, this method will
    generate an example per field. For example, if there are two fields on which
    to perform replacement, the method will perform replacement first on one
    field to produce an example (other fields left intact), and then perform
    replacement on the second field (again copying all other fields from the
    original datum).

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

    ignore_casing = config.get(IGNORE_CASING_KEY, True)
    subs_string = config.get(SUBSTITUTIONS_KEY, None)
    if subs_string:
      replacements = self.parse_subs_string(
          subs_string, ignore_casing=ignore_casing)
    else:
      replacements = self.default_replacements

    # If replacements dictionary is empty, do not attempt to match.
    if not replacements:
      return []

    replacement_regex = self._get_replacement_pattern(
        replacements, ignore_casing=ignore_casing)

    # If config key is missing, generate no examples.
    fields_to_replace = list(config.get(FIELDS_TO_REPLACE_KEY, []))
    if not fields_to_replace:
      return []

    # TODO(lit-dev): move this to generate_all(), so we read the spec once
    # instead of on every example.
    text_keys = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    if not text_keys:
      return []

    text_keys = [key for key in text_keys if key in fields_to_replace]

    new_examples = []
    for text_key in text_keys:
      text_data = example[text_key]
      for new_val in self.generate_counterfactuals(
          text_data, replacement_regex, replacements,
          ignore_casing=ignore_casing):
        new_example = copy.deepcopy(example)
        new_example[text_key] = new_val
        new_examples.append(new_example)

    return new_examples

  def config_spec(self) -> types.Spec:
    return {
        # Requires a substitution string. Include a default.
        SUBSTITUTIONS_KEY: types.TextSegment(default='great -> terrible'),
        FIELDS_TO_REPLACE_KEY:
            types.MultiFieldMatcher(
                spec='input',
                types=['TextSegment'],
                select_all=True),
    }
