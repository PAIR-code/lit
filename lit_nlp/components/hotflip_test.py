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
"""Tests for lit_nlp.components.hotflip."""

from absl.testing import absltest
from lit_nlp.api import types as lit_types
from lit_nlp.components import hotflip
# TODO(lit-dev): Move glue_models out of lit_nlp/examples
from lit_nlp.examples.models import glue_models
import numpy as np


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
extract_compressed_file=True)


class ModelBasedHotflipTest(absltest.TestCase):

  def setUp(self):
    super(ModelBasedHotflipTest, self).setUp()
    self.hotflip = hotflip.HotFlip()
    self.model = glue_models.SST2Model(BERT_TINY_PATH)
    self.pred_key = 'probas'
    self.config = {hotflip.PREDICTION_KEY: self.pred_key}

  def test_find_fields(self):
    fields = self.hotflip.find_fields(self.model.output_spec(),
                                      lit_types.MulticlassPreds)
    self.assertEqual(['probas'], fields)
    fields = self.hotflip.find_fields(self.model.output_spec(),
                                      lit_types.TokenGradients,
                                      'tokens_sentence')
    self.assertEqual(['token_grad_sentence'], fields)

  def test_find_fields_empty(self):
    fields = self.hotflip.find_fields(self.model.output_spec(),
                                      lit_types.TokenGradients,
                                      'input_embs_sentence')
    self.assertEmpty(fields)

  def test_hotflip_num_ex(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.config[hotflip.NUM_EXAMPLES_KEY] = 0
    self.assertEmpty(
        self.hotflip.generate(ex, self.model, None, self.config))
    self.config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.assertLen(
        self.hotflip.generate(ex, self.model, None, self.config), 1)
    self.config[hotflip.NUM_EXAMPLES_KEY] = 2
    self.assertLen(
        self.hotflip.generate(ex, self.model, None, self.config), 2)

  def test_hotflip_freeze_tokens(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.config[hotflip.NUM_EXAMPLES_KEY] = 10
    self.config[hotflip.TOKENS_TO_IGNORE_KEY] = ['terrible']
    generated = self.hotflip.generate(
        ex, self.model, None, self.config)
    for gen in generated:
      self.assertLen(gen['tokens_sentence'], 6)
      self.assertEqual('terrible', gen['tokens_sentence'][4])

    self.config[hotflip.NUM_EXAMPLES_KEY] = 10
    self.config[hotflip.TOKENS_TO_IGNORE_KEY] = ['terrible', 'long']
    generated = self.hotflip.generate(
        ex, self.model, None, self.config)
    for gen in generated:
      self.assertEqual('long', gen['tokens_sentence'][1])
      self.assertEqual('terrible', gen['tokens_sentence'][4])

  def test_hotflip_drops(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.config[hotflip.DROP_TOKENS_KEY] = True
    generated = self.hotflip.generate(
        ex, self.model, None, self.config)
    self.assertLess(len(generated[0]['tokens_sentence']), 6)

  def test_hotflip_max_flips(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.config[hotflip.MAX_FLIPS_KEY] = 1
    generated = self.hotflip.generate(
        ex, self.model, None, self.config)
    self.assertLen(generated, 1)

    num_flipped = 0
    pred = list(self.model.predict([ex]))[0]
    pred_tokens = pred['tokens_sentence']
    gen_tokens = generated[0]['tokens_sentence']
    for i in range(len(gen_tokens)):
      if gen_tokens[i] != pred_tokens[i]:
        num_flipped += 1
    self.assertEqual(1, num_flipped)

    ex = {'sentence': 'this long movie is terrible and horrible.'}
    self.config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.config[hotflip.MAX_FLIPS_KEY] = 1
    generated = self.hotflip.generate(
        ex, self.model, None, self.config)
    self.assertEmpty(generated)

  def test_hotflip_changes_pred(self):
    ex = {'sentence': 'this long movie is terrible.'}
    pred = list(self.model.predict([ex]))[0]
    pred_class = str(np.argmax(pred['probas']))
    self.assertEqual('0', pred_class)
    generated = self.hotflip.generate(ex, self.model, None, self.config)
    for gen in generated:
      self.assertEqual('1', gen['label'])

  def test_hotflip_fails_without_pred_key(self):
    ex = {'sentence': 'this long movie is terrible.'}
    with self.assertRaises(AssertionError):
      self.hotflip.generate(ex, self.model, None, None)


if __name__ == '__main__':
  absltest.main()
