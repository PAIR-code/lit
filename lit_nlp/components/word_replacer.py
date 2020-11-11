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
from typing import Dict, Iterator, List, Text, Tuple, Optional, Pattern

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict
# Patterns to match regex lookaround patterns at the beginning/end of a string.
# Lookbehinds are either "(?<=WORD)" or "(?<!STRING)".
lookbehind_re = re.compile(r'^(\(\?<[=!][\w\s\b]+\))|$')
# Lookaheads are either "(?=WORD)" or "(?!STRING)".
lookahead_re = re.compile(r'(\(\?[=!][\w\s\b]+\))$|$')


class WordReplacer(lit_components.Generator):
  """Word replacement generator."""

  def __init__(self, replacements: Optional[Dict[Text, List[Text]]] = None):
    # Populate dictionary with replacement options.
    if replacements is not None:
      assert isinstance(replacements, dict), 'Replacements must be a dict.'
      assert all([isinstance(tgt, list) for tgt in replacements.values()
                 ]), 'Replacement dict must be Text->List[Text]'
      self.default_replacements = replacements
    else:
      self.default_replacements = {}

  def parse_subs_string(self,
                        subs_string: Text,
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
      # Remove lookaround info (e.g., "(?<=a )test" -> "test").
      replacements[src.strip()] = [tgt.strip() for tgt in tgts]
    return replacements

  def _trim_lookarounds(self, pattern_string: str) -> str:
    """Trims lookaround regex from the matching string."""
    pattern_string = lookbehind_re.sub('', pattern_string)
    pattern_string = lookahead_re.sub('', pattern_string)
    return pattern_string

  def _split_lookarounds_from_pattern(
      self, pattern_string: str) -> Tuple[str, str, str]:
    """Extracts the lookarounds from a raw string and returns it in parts."""
    # Extract lookaround pattern if exists, otherwise empty string.
    lookbehind = lookbehind_re.findall(pattern_string)[0]
    lookahead = lookahead_re.findall(pattern_string)[0]
    # Remove lookaround pattern from pattern string.
    pattern_string = self._trim_lookarounds(pattern_string)
    return lookbehind, pattern_string, lookahead

  def _clean_lookarounds_from_replacements(
      self, replacements: Dict[Text, List[Text]]) -> Dict[Text, List[Text]]:
    """trim lookarounds from replacement indices to match re.match value."""
    clean_replacements = {
        self._trim_lookarounds(key): value
        for key, value in replacements.items()
    }
    return clean_replacements

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
    In case that lookarounds (e.g., "(?<=dog )") exists, no punctuation is added
    since that would break the logic and we treat the example as one that
    includes punctuation.

    Args:
      replacements: A dict of word replacements
      ignore_casing: Ignore casing for source words if True.

    Returns:
      regexp_pattern: Compiled regexp pattern used to find source words in
                      replacements.
    """
    re_strings = []
    for s in replacements:
      lookbehind, s, lookahead = self._split_lookarounds_from_pattern(s)
      pattern_str = r'\b%s\b' % re.escape(s)
      # If the source word ends or begins with a non-word character (see above.)
      # Ignore for lookarounds since those matches fail in the presence of \b.
      if (not re.search(pattern_str, s)) or lookbehind or lookahead:
        pattern_str = r'%s' % re.escape(s)
      pattern_str = f'{lookbehind}{pattern_str}{lookahead}'
      re_strings.append(pattern_str)
    casing_flag = re.IGNORECASE if ignore_casing else 0

    patterns = re.compile('|'.join(re_strings), casing_flag)
    return patterns

  def generate_counterfactuals(self,
                               text: Text,
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
    """Replace words based on replacement list."""
    del model  # Unused.
    ignore_casing = config.get('ignore_casing', True) if config else True
    subs_string = config.get('Substitutions') if config else None
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
    new_examples = []
    # TODO(lit-dev): move this to generate_all(), so we read the spec once
    # instead of on every example.
    text_keys = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    # Remove regex fragments from the actual string that is being replaced.
    clean_replacements = self._clean_lookarounds_from_replacements(replacements)
    for text_key in text_keys:
      text_data = example[text_key]
      for new_val in self.generate_counterfactuals(
          text_data,
          replacement_regex,
          clean_replacements,
          ignore_casing=ignore_casing):
        new_example = copy.deepcopy(example)
        new_example[text_key] = new_val
        new_examples.append(new_example)

    return new_examples

  def spec(self) -> types.Spec:
    return {
        # Requires a substitution string. Include a default.
        'Substitutions': types.TextSegment(default='great -> terrible')
    }
