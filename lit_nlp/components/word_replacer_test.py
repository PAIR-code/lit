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
"""Tests for lit_nlp.generators.word_replacer."""

from absl.testing import absltest
from absl.testing import parameterized

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import word_replacer
from lit_nlp.lib import testing_utils


class WordReplacerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_spec: lit_types.Spec = {'text': lit_types.TextSegment()}
    self.model = testing_utils.RegressionModelForTesting(test_spec)
    # Dataset is only used for spec in word_replacer so define once
    self.dataset = lit_dataset.Dataset(
        spec=test_spec, examples=[{'text': 'blank'}]
    )
    self.generator = word_replacer.WordReplacer()

  def test_default_replacements(self):
    example = {'text': 'xyz yzy zzz.'}
    config = {word_replacer.FIELDS_TO_REPLACE_KEY: ['text']}
    generated = self.generator.generate(
        example, self.model, self.dataset, config
    )
    self.assertEqual(generated, [])

  def test_init_replacements(self):
    generator = word_replacer.WordReplacer(replacements={'tree': ['car']})
    example = {'text': 'black truck hit the tree'}
    config = {word_replacer.FIELDS_TO_REPLACE_KEY: ['text']}
    generated = generator.generate(example, self.model, self.dataset, config)
    self.assertEqual(generated, [{'text': 'black truck hit the car'}])

  @parameterized.named_parameters(
      dict(
          testcase_name='ascii_to_ascii_ignore_caps',
          example={'text': 'Capitalization is ignored.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'capitalization -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'blank is ignored.'}],
      ),
      dict(
          testcase_name='ascii_to_ascii_respect_caps_change',
          example={'text': 'Capitalization is ignored.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'Capitalization -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
              word_replacer.IGNORE_CASING_KEY: False,
          },
          expected=[{'text': 'blank is ignored.'}],
      ),
      dict(
          testcase_name='ascii_to_ascii_respect_caps_no_change',
          example={'text': 'Capitalization is ignored.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'capitalization -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
              word_replacer.IGNORE_CASING_KEY: False,
          },
          expected=[],
      ),
      dict(
          testcase_name='deletion',
          example={'text': 'A storm is raging.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'storm -> ',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'A  is raging.'}],
      ),
      dict(
          testcase_name='multiple_targets',
          example={'text': 'It`s raining cats and dogs.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'dogs -> horses|donkeys',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[
              {'text': 'It`s raining cats and horses.'},
              {'text': 'It`s raining cats and donkeys.'}
          ],
      ),
      dict(
          testcase_name='multiple_words',
          example={'text': 'A red cat is coming.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'red cat -> black dog',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'A black dog is coming.'}],
      ),
      dict(
          testcase_name='no_partial_match',
          example={'text': 'A catastrophic storm'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'cat -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[],
      ),
      dict(
          testcase_name='special_chars_punctuation',
          example={'text': 'A catastrophic storm .'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: '. -> -',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'A catastrophic storm -'}],
      ),
      dict(
          testcase_name='special_chars_repeated_punctuation',
          example={'text': 'A.catastrophic. storm'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: '. -> -',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[
              {'text': 'A-catastrophic. storm'},
              {'text': 'A.catastrophic- storm'},
          ],
      ),
      dict(
          testcase_name='special_chars_repeated_multichar_punctuation',
          example={'text': 'A...catastrophic.... storm'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: '.. -> --',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[
              {'text': 'A--.catastrophic.... storm'},
              {'text': 'A...catastrophic--.. storm'},
              {'text': 'A...catastrophic..-- storm'},
          ],
      ),
      dict(
          testcase_name='special_chars_underscore',
          example={'text': 'A nasty_storm is raging.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'nasty_storm -> nice_storm',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'A nice_storm is raging.'}],
      ),
      dict(
          testcase_name='two_repetitions_yields_two_examples',
          example={'text': 'maybe repetition repetition maybe'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'repetition -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[
              {'text': 'maybe blank repetition maybe'},
              {'text': 'maybe repetition blank maybe'},
          ],
      ),
      dict(
          testcase_name='unicode_latin_to_ascii',
          example={'text': 'Is répertoire a unicode word?'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'répertoire -> repertoire',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'Is repertoire a unicode word?'}],
      ),
      dict(
          testcase_name='unicode_pictograph_to_unicode_pictograph',
          example={'text': '♞ is a black chess knight.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: '♞ -> ♟',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': '♟ is a black chess knight.'}],
      ),
      dict(
          testcase_name='words_with_and_near_punctuation',
          example={'text': 'It`s raining cats and dogs.'},
          config={
              word_replacer.SUBSTITUTIONS_KEY: 'dogs -> blank',
              word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
          },
          expected=[{'text': 'It`s raining cats and blank.'}],
      ),
  )
  def test_replacement(
      self,
      example: lit_types.JsonDict,
      config: lit_types.JsonDict,
      expected: list[lit_types.JsonDict]
  ):
    generated = self.generator.generate(
        example, self.model, self.dataset, config
    )
    self.assertCountEqual(generated, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='multiple_replacements_all_valid',
          query_string='foo -> bar, spam -> eggs',
          expected={'foo': ['bar'], 'spam': ['eggs']},
      ),
      dict(
          testcase_name='multiple_replacements_ignore_malformed',
          query_string='foo -> bar, spam eggs',
          expected={'foo': ['bar']},
      ),
      dict(
          testcase_name='multiple_replacements_mulitple_targets',
          query_string='foo -> bar, spam -> eggs|donuts | cream',
          expected={'foo': ['bar'], 'spam': ['eggs', 'donuts', 'cream']},
      ),
      dict(
          testcase_name='empty',
          query_string='',
          expected={},
      ),
      dict(
          testcase_name='single_unicode_replacement',
          query_string='♞ -> ♟',
          expected={'♞': ['♟']},
      )
  )
  def test_parse_sub_string(
      self, query_string: str, expected: dict[str, list[str]]
  ):
    parsed = self.generator.parse_subs_string(query_string)
    self.assertEqual(parsed, expected)

if __name__ == '__main__':
  absltest.main()
