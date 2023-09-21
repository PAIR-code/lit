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
"""Word replacement generator."""

from collections.abc import Iterator, Sequence
import re
from typing import Optional

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

_JsonDict = types.JsonDict

IGNORE_CASING_KEY = 'ignore_casing'
SUBSTITUTIONS_KEY = 'Substitutions'
FIELDS_TO_REPLACE_KEY = 'Fields to replace'


class WordReplacer(lit_components.Generator):
  """Generate new examples by replacing words in examples.

  Substitutions must be of the form 'foo -> bar, spam -> eggs'.
  """

  def __init__(self, replacements: Optional[dict[str, list[str]]] = None):
    # Populate dictionary with replacement options.
    if replacements is not None:
      assert isinstance(replacements, dict), 'Replacements must be a dict.'
      assert all([isinstance(tgt, list) for tgt in replacements.values()
                 ]), 'Replacement dict must be str->list[str]'
      self.default_replacements = replacements
    else:
      self.default_replacements = {}

  def parse_subs_string(self, subs_string: str,
                        ignore_casing: bool = True) -> dict[str, list[str]]:
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
                               replacements: dict[str, list[str]],
                               ignore_casing: bool = True) -> re.Pattern[str]:
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

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del model  # Unused by WordReplacer
    return utils.spec_contains(dataset.spec(), types.TextSegment)

  def generate_counterfactuals(
      self, text: str,
      replacement_regex: re.Pattern[str],
      replacements: dict[str, list[str]],
      ignore_casing: bool = True) -> Iterator[str]:
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

  def generate_all(self,
                   inputs: list[_JsonDict],
                   model: lit_model.Model,
                   dataset: lit_dataset.Dataset,
                   config: Optional[_JsonDict] = None) -> list[list[_JsonDict]]:
    """Run generation on a set of inputs.

    Args:
      inputs: sequence of inputs, following model.input_spec()
      model: optional model to use to generate new examples.
      dataset: optional dataset which the current examples belong to.
      config: optional runtime config.

    Returns:
      list of list of new generated inputs, following model.input_spec()
    """
    config: _JsonDict = config or {}
    text_fields_to_replace = self._compute_fields_to_replace(dataset, config)
    return [
        self.generate(ex, model, dataset, config, text_fields_to_replace)
        for ex in inputs
    ]

  def generate(
      self,
      example: _JsonDict,
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      config: Optional[_JsonDict] = None,
      text_fields_to_replace: Optional[Sequence[str]] = None,
  ) -> list[_JsonDict]:
    """Replace words based on replacement list.

    Note: If multiple fields are selected for replacement, this method will
    generate an example per field. For example, if there are two fields on which
    to perform replacement, the method will perform replacement first on one
    field to produce an example (other fields left intact), and then perform
    replacement on the second field (again copying all other fields from the
    original datum).

    Args:
      example: the example used for basis of generated examples.
      model: unused.
      dataset: the dataset.
      config: user-provided config properties.
      text_fields_to_replace: Opionally the fields over which the replacer runs.

    Returns:
      examples: a list of generated examples.
    """
    del model  # Unused.

    config: _JsonDict = config or {}

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
    if text_fields_to_replace is None:
      text_fields_to_replace = self._compute_fields_to_replace(
          dataset, config
      )
    if not text_fields_to_replace:
      return []

    new_examples = []
    for text_field in text_fields_to_replace:
      for new_val in self.generate_counterfactuals(
          example[text_field],
          replacement_regex,
          replacements,
          ignore_casing=ignore_casing,
      ):
        new_examples.append(
            utils.make_modified_input(
                example, {text_field: new_val}, 'WordReplacer'
            )
        )

    return new_examples

  def _compute_fields_to_replace(
      self, dataset: lit_dataset.Dataset, config: _JsonDict
  ) -> Sequence[str]:
    fields_to_replace: tuple[str] = tuple(config.get(FIELDS_TO_REPLACE_KEY, []))
    if not fields_to_replace:
      return []

    text_fields = utils.find_spec_keys(dataset.spec(), types.TextSegment)
    return [key for key in text_fields if key in fields_to_replace]

  def config_spec(self) -> types.Spec:
    return {
        # Requires a substitution string. Include a default.
        SUBSTITUTIONS_KEY: types.TextSegment(default='great -> terrible'),
        FIELDS_TO_REPLACE_KEY: types.MultiFieldMatcher(
            spec='input', types=['TextSegment'], select_all=True
        ),
    }
