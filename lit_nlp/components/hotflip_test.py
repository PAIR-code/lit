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
from lit_nlp.lib import utils
import numpy as np


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
STSB_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/stsb_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
  extract_compressed_file=True)
STSB_PATH = transformers.file_utils.cached_path(STSB_PATH,
  extract_compressed_file=True)


class SSTModelWithoutEmbeddings(glue_models.SST2Model):

  def get_embedding_table(self):
    raise NotImplementedError('get_embedding_table() not implemented for ' +
                              self.__class__.__name__)


class SSTModelWithoutTokens(glue_models.SST2Model):

  def input_spec(self):
    ret = glue_models.STSBModel.input_spec(self)
    token_keys = utils.find_spec_keys(ret, lit_types.Tokens)
    for k in token_keys:
      ret.pop(k, None)
    return ret


class SSTModelWithoutGradients(glue_models.SST2Model):

  def output_spec(self):
    ret = glue_models.STSBModel.output_spec(self)
    token_gradient_keys = utils.find_spec_keys(ret, lit_types.TokenGradients)
    for k in token_gradient_keys:
      ret.pop(k, None)
    return ret


class ModelBasedHotflipTest(absltest.TestCase):

  def setUp(self):
    super(ModelBasedHotflipTest, self).setUp()
    self.hotflip = hotflip.HotFlip()

    # Classification model that clasifies a given input sentence.
    self.classification_model = glue_models.SST2Model(BERT_TINY_PATH)
    self.classification_config = {hotflip.PREDICTION_KEY: 'probas'}

    # A wrapped version of the classification model that does not expose
    # embeddings.
    self.classification_model_without_embeddings = SSTModelWithoutEmbeddings(
        BERT_TINY_PATH)

    # A wrapped version of the classification model that does not take tokens
    # as input.
    self.classification_model_without_tokens = SSTModelWithoutTokens(
        BERT_TINY_PATH)

    # A wrapped version of the classification model that does not expose
    # gradients.
    self.classification_model_without_gradients = SSTModelWithoutGradients(
        BERT_TINY_PATH)

    # Regression model determining similarity between two input sentences.
    self.regression_model = glue_models.STSBModel(STSB_PATH)
    self.regression_config = {hotflip.PREDICTION_KEY: 'score'}

  def test_find_fields(self):
    fields = self.hotflip.find_fields(self.classification_model.output_spec(),
                                      lit_types.MulticlassPreds)
    self.assertEqual(['probas'], fields)
    fields = self.hotflip.find_fields(self.classification_model.output_spec(),
                                      lit_types.TokenGradients,
                                      'tokens_sentence')
    self.assertEqual(['token_grad_sentence'], fields)

  def test_find_fields_empty(self):
    fields = self.hotflip.find_fields(self.classification_model.output_spec(),
                                      lit_types.TokenGradients,
                                      'input_embs_sentence')
    self.assertEmpty(fields)

  def test_hotflip_num_ex(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 0
    self.assertEmpty(
        self.hotflip.generate(ex, self.classification_model, None,
                              self.classification_config))
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.assertLen(
        self.hotflip.generate(ex, self.classification_model, None,
                              self.classification_config), 1)
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 2
    self.assertLen(
        self.hotflip.generate(ex, self.classification_model, None,
                              self.classification_config), 2)

  def test_hotflip_num_ex_multi_input(self):
    ex = {'sentence1': 'this long movie is terrible.',
          'sentence2': 'this short movie is great.'}
    self.regression_config[hotflip.NUM_EXAMPLES_KEY] = 2
    thresh = 2
    self.regression_config[hotflip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence1',
        'tokens_sentence2',
    ]
    self.assertLen(
        self.hotflip.generate(ex, self.regression_model, None,
                              self.regression_config), 2)

  def test_hotflip_freeze_tokens(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 10
    self.classification_config[hotflip.TOKENS_TO_IGNORE_KEY] = ['terrible']
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    cfs = self.hotflip.generate(
        ex, self.classification_model, None, self.classification_config)
    for cf in cfs:
      tokens = cf['tokens_sentence']
      self.assertLen(tokens, 6)
      self.assertEqual('terrible', tokens[4])

    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 10
    self.classification_config[hotflip.TOKENS_TO_IGNORE_KEY] = ['long',
                                                                'terrible']
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    cfs = self.hotflip.generate(
        ex, self.classification_model, None, self.classification_config)
    for cf in cfs:
      tokens = cf['tokens_sentence']
      self.assertEqual('terrible', tokens[4])
      self.assertEqual('long', tokens[1])

  def test_hotflip_freeze_tokens_multi_input(self):
    ex = {'sentence1': 'this long movie is terrible.',
          'sentence2': 'this long movie is great.'}
    self.regression_config[hotflip.NUM_EXAMPLES_KEY] = 10
    thresh = 2
    self.regression_config[hotflip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[hotflip.TOKENS_TO_IGNORE_KEY] = ['long', 'terrible']
    self.regression_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence1',
        'tokens_sentence2',
    ]
    cfs = self.hotflip.generate(ex, self.regression_model, None,
                                self.regression_config)
    for cf in cfs:
      tokens1 = cf['tokens_sentence1']
      tokens2 = cf['tokens_sentence2']
      self.assertEqual('terrible', tokens1[4])
      self.assertEqual('long', tokens1[1])
      self.assertEqual('long', tokens2[1])

  def test_hotflip_max_flips(self):
    ex = {'sentence': 'this long movie is terrible.'}
    ex_output = list(self.classification_model.predict([ex]))[0]
    ex_tokens = ex_output['tokens_sentence']

    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[hotflip.MAX_FLIPS_KEY] = 1
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    cfs = self.hotflip.generate(
        ex, self.classification_model, None, self.classification_config)
    cf_tokens = list(cfs)[0]['tokens_sentence']
    self.assertEqual(1, sum([1 for i, t in enumerate(cf_tokens)
                             if t != ex_tokens[i]]))

    ex = {'sentence': 'this long movie is terrible and horrible.'}
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[hotflip.MAX_FLIPS_KEY] = 1
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    cfs = self.hotflip.generate(
        ex, self.classification_model, None, self.classification_config)
    self.assertEmpty(cfs)

  def test_hotflip_max_flips_multi_input(self):
    ex = {'sentence1': 'this long movie is terrible.',
          'sentence2': 'this short movie is great.'}
    ex_output = list(self.regression_model.predict([ex]))[0]
    ex_tokens1 = ex_output['tokens_sentence1']
    ex_tokens2 = ex_output['tokens_sentence2']

    self.regression_config[hotflip.NUM_EXAMPLES_KEY] = 20
    thresh = 2
    self.regression_config[hotflip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[hotflip.MAX_FLIPS_KEY] = 1
    self.regression_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence1',
        'tokens_sentence2',
    ]
    cfs = self.hotflip.generate(ex, self.regression_model, None,
                                self.regression_config)
    for cf in cfs:
      # Number of flips in each field should be no more than MAX_FLIPS.
      cf_tokens1 = cf['tokens_sentence1']
      cf_tokens2 = cf['tokens_sentence2']
      self.assertLessEqual(sum([1 for i, t in enumerate(cf_tokens1)
                                if t != ex_tokens1[i]]), 1)
      self.assertLessEqual(sum([1 for i, t in enumerate(cf_tokens2)
                                if t != ex_tokens2[i]]), 1)

  def test_hotflip_only_flip_one_field(self):
    ex = {'sentence1': 'this long movie is terrible.',
          'sentence2': 'this short movie is great.'}
    self.regression_config[hotflip.NUM_EXAMPLES_KEY] = 10
    thresh = 2
    self.regression_config[hotflip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence1',
        'tokens_sentence2',
    ]
    cfs = self.hotflip.generate(ex, self.regression_model, None,
                                self.regression_config)
    for cf in cfs:
      self.assertTrue(
          (cf['sentence1'] == ex['sentence1']) or
          (cf['sentence2'] == ex['sentence2']))

  def test_hotflip_changes_pred_class(self):
    ex = {'sentence': 'this long movie is terrible.'}
    ex_output = list(self.classification_model.predict([ex]))[0]
    pred_class = str(np.argmax(ex_output['probas']))
    self.assertEqual('0', pred_class)
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    cfs = self.hotflip.generate(ex, self.classification_model, None,
                                self.classification_config)
    cf_outputs = self.classification_model.predict(cfs)
    for cf_output in cf_outputs:
      self.assertNotEqual(np.argmax(ex_output['probas']),
                          np.argmax(cf_output['probas']))

  def test_hotflip_changes_regression_score(self):
    ex = {'sentence1': 'this long movie is terrible.',
          'sentence2': 'this short movie is great.'}
    self.regression_config[hotflip.NUM_EXAMPLES_KEY] = 2
    ex_output = list(self.regression_model.predict([ex]))[0]
    thresh = 2
    self.regression_config[hotflip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence1',
        'tokens_sentence2',
    ]
    cfs = self.hotflip.generate(ex, self.regression_model, None,
                                self.regression_config)
    cf_outputs = self.regression_model.predict(cfs)
    for cf_output in cf_outputs:
      self.assertNotEqual((ex_output['score'] <= thresh),
                          (cf_output['score'] <= thresh))

  def test_hotflip_fails_without_embeddings(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    with self.assertRaises(NotImplementedError):
      self.hotflip.generate(ex, self.classification_model_without_embeddings,
                            None, self.classification_config)

  def test_hotflip_fails_without_tokens(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    with self.assertRaises(AssertionError):
      self.hotflip.generate(ex, self.classification_model_without_tokens,
                            None, self.classification_config)

  def test_hotflip_fails_without_gradients(self):
    ex = {'sentence': 'this long movie is terrible.'}
    self.classification_config[hotflip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[hotflip.FIELDS_TO_HOTFLIP_KEY] = [
        'tokens_sentence',
    ]
    with self.assertRaises(AssertionError):
      self.hotflip.generate(ex, self.classification_model_without_gradients,
                            None, self.classification_config)

  def test_hotflip_fails_without_pred_key(self):
    ex = {'sentence': 'this long movie is terrible.'}
    with self.assertRaises(AssertionError):
      self.hotflip.generate(ex, self.classification_model, None, None)


if __name__ == '__main__':
  absltest.main()
