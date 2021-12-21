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

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import word_replacer
from lit_nlp.lib import testing_utils


class WordReplacerTest(absltest.TestCase):

  def test_all_replacements(self):
    input_spec = {'text': lit_types.TextSegment()}
    model = testing_utils.TestRegressionModel(input_spec)
    # Dataset is only used for spec in word_replacer so define once
    dataset = lit_dataset.Dataset(input_spec, [{'text': 'blank'}])

    ## Test replacements
    generator = word_replacer.WordReplacer()
    # Unicode to Unicode
    input_dict = {'text': '♞ is a black chess knight.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: '♞ -> ♟',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': '♟ is a black chess knight.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Unicode to ASCII
    input_dict = {'text': 'Is répertoire a unicode word?'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'répertoire -> repertoire',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'Is repertoire a unicode word?'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Ignore capitalization
    input_dict = {'text': 'Capitalization is ignored.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'Capitalization -> blank',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'blank is ignored.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    input_dict = {'text': 'Capitalization is ignored.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'capitalization -> blank',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'blank is ignored.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Do not Ignore capitalization
    input_dict = {'text': 'Capitalization is important.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'Capitalization -> blank',
        word_replacer.IGNORE_CASING_KEY: False,
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'blank is important.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    input_dict = {'text': 'Capitalization is important.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'capitalization -> blank',
        word_replacer.IGNORE_CASING_KEY: False,
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = []
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Repetition
    input_dict = {'text': 'maybe repetition repetition maybe'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'repetition -> blank',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'maybe blank repetition maybe'},
                {'text': 'maybe repetition blank maybe'}]
    self.assertCountEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # No partial match
    input_dict = {'text': 'A catastrophic storm'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'cat -> blank',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = []
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    ## Special characters
    # Punctuation
    input_dict = {'text': 'A catastrophic storm .'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: '. -> -',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A catastrophic storm -'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    input_dict = {'text': 'A.catastrophic. storm'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: '. -> -',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A-catastrophic. storm'},
                {'text': 'A.catastrophic- storm'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    input_dict = {'text': 'A...catastrophic.... storm'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: '.. -> --',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A--.catastrophic.... storm'},
                {'text': 'A...catastrophic--.. storm'},
                {'text': 'A...catastrophic..-- storm'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Underscore
    input_dict = {'text': 'A catastrophic_storm is raging.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'catastrophic_storm -> nice_storm',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A nice_storm is raging.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Deletion
    input_dict = {'text': 'A storm is raging.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'storm -> ',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A  is raging.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Word next to punctuation and words with punctuation.
    input_dict = {'text': 'It`s raining cats and dogs.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'dogs -> blank',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'It`s raining cats and blank.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    # Multiple target tokens.
    input_dict = {'text': 'It`s raining cats and dogs.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'dogs -> horses|donkeys',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'It`s raining cats and horses.'},
                {'text': 'It`s raining cats and donkeys.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    ## Test default_replacements applied at init.
    replacements = {'tree': ['car']}
    generator = word_replacer.WordReplacer(replacements=replacements)
    input_dict = {'text': 'black truck hit the tree'}
    expected = [{'text': 'black truck hit the car'}]
    config_dict = {
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }

    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

    ## Test not passing replacements not breaking.
    generator = word_replacer.WordReplacer()
    input_dict = {'text': 'xyz yzy zzz.'}
    expected = []

    self.assertEqual(
        generator.generate(input_dict, model, dataset), expected)

    # Multi word match.
    input_dict = {'text': 'A red cat is coming.'}
    config_dict = {
        word_replacer.SUBSTITUTIONS_KEY: 'red cat -> black dog',
        word_replacer.FIELDS_TO_REPLACE_KEY: ['text'],
    }
    expected = [{'text': 'A black dog is coming.'}]
    self.assertEqual(
        generator.generate(input_dict, model, dataset, config=config_dict),
        expected)

  def test_parse_sub_string(self):
    generator = word_replacer.WordReplacer()

    query_string = 'foo -> bar, spam -> eggs'
    expected = {'foo': ['bar'], 'spam': ['eggs']}
    self.assertDictEqual(generator.parse_subs_string(query_string), expected)

    # Should ignore the malformed rule
    query_string = 'foo -> bar, spam eggs'
    expected = {'foo': ['bar']}
    self.assertDictEqual(generator.parse_subs_string(query_string), expected)

    # Multiple target tokens.
    query_string = 'foo -> bar, spam -> eggs|donuts | cream'
    expected = {'foo': ['bar'], 'spam': ['eggs', 'donuts', 'cream']}
    self.assertDictEqual(generator.parse_subs_string(query_string), expected)

    query_string = ''
    expected = {}
    self.assertDictEqual(generator.parse_subs_string(query_string), expected)

    query_string = '♞ -> ♟'
    expected = {'♞': ['♟']}
    self.assertDictEqual(generator.parse_subs_string(query_string), expected)

if __name__ == '__main__':
  absltest.main()
