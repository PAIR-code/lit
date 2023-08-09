# Copyright 2023 Google LLC
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
"""Tests for lit_nlp.components.scrambler."""


from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import scrambler
from lit_nlp.lib import testing_utils


class ScramblerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_spec: lit_types.Spec = {
        'sentence': lit_types.TextSegment(),
        '_id': lit_types.TextSegment(),
        '_meta': lit_types.JsonDict,
    }
    self.model = testing_utils.RegressionModelForTesting(test_spec)
    # Dataset is only used for spec in Scrambler so define once
    self.dataset = lit_dataset.Dataset(
        spec=test_spec,
        examples=[{'text': 'blank', '_id': 'a1b2c3', '_meta': {}}],
    )
    self.generator = scrambler.Scrambler()

  def test_scrambler_id_changes(self):
    example = {
        'sentence': 'this is the sentence to be scrambled',
        '_id': 'a1b2c3',
        '_meta': {'parentId': '000000'},
    }

    config = {scrambler.FIELDS_TO_SCRAMBLE_KEY: ['sentence']}
    generated_example = self.generator.generate(
        example, self.model, self.dataset, config
    )[0]
    self.assertNotEqual(generated_example, example)
    self.assertNotEqual(generated_example['_id'], example['_id'])
    self.assertEqual(
        set(generated_example['sentence'].split()),
        set(example['sentence'].split()),
    )
    self.assertEqual(generated_example['_meta']['parentId'], example['_id'])


if __name__ == '__main__':
  absltest.main()
