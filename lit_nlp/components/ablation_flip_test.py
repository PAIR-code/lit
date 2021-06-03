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
"""Tests for lit_nlp.components.ablation_flip."""

from absl.testing import absltest
from lit_nlp.components import ablation_flip
# TODO(lit-dev): Move glue_models out of lit_nlp/examples
from lit_nlp.examples.models import glue_models
import numpy as np


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
STSB_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/stsb_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
  extract_compressed_file=True)
STSB_PATH = transformers.file_utils.cached_path(STSB_PATH,
  extract_compressed_file=True)


class ModelBasedAblationFlipTest(absltest.TestCase):

  def setUp(self):
    super(ModelBasedAblationFlipTest, self).setUp()
    self.ablation_flip = ablation_flip.AblationFlip()

    # Classification model that clasifies a given input sentence.
    self.classification_model = glue_models.SST2Model(BERT_TINY_PATH)
    self.classification_config = {ablation_flip.PREDICTION_KEY: 'probas'}

    # Regression model determining similarity between two input sentences.
    self.regression_model = glue_models.STSBModel(STSB_PATH)
    self.regression_config = {ablation_flip.PREDICTION_KEY: 'score'}

  def test_ablation_flip_num_ex(self):
    ex = {'sentence': 'this long movie was terrible'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 0
    self.assertEmpty(
        self.ablation_flip.generate(ex, self.classification_model, None,
                                    self.classification_config))
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.assertLen(
        self.ablation_flip.generate(ex, self.classification_model, None,
                                    self.classification_config), 1)
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 2
    self.assertLen(
        self.ablation_flip.generate(ex, self.classification_model, None,
                                    self.classification_config), 2)

  def test_ablation_flip_num_ex_multi_input(self):
    ex = {'sentence1': 'this long movie is terrible',
          'sentence2': 'this short movie is great'}
    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 2
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    self.assertLen(
        self.ablation_flip.generate(ex, self.regression_model, None,
                                    self.regression_config), 2)

  def test_ablation_flip_freeze_tokens(self):
    ex = {'sentence': 'this long movie is terrible'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 10
    tokens_to_freeze = ['long', 'terrible']
    self.classification_config[
        ablation_flip.TOKENS_TO_IGNORE_KEY] = tokens_to_freeze
    cfs = self.ablation_flip.generate(ex, self.classification_model, None,
                                      self.classification_config)
    for cf in cfs:
      tokens = self.ablation_flip.tokenize(cf['sentence'])
      self.assertContainsSubset(tokens_to_freeze, tokens)

  def test_ablation_flip_freeze_tokens_multi_input(self):
    ex = {'sentence1': 'this long movie is terrible',
          'sentence2': 'this long movie is great'}
    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 10
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[ablation_flip.TOKENS_TO_IGNORE_KEY] = [
        'long', 'terrible'
    ]
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    for cf in cfs:
      tokens1 = self.ablation_flip.tokenize(cf['sentence1'])
      tokens2 = self.ablation_flip.tokenize(cf['sentence2'])
      self.assertContainsSubset(['long', 'terrible'], tokens1)
      self.assertContainsSubset(['long'], tokens2)

  def test_ablation_flip_max_ablations(self):
    ex = {'sentence': 'this movie is terrible'}
    ex_tokens = self.ablation_flip.tokenize(ex['sentence'])
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[ablation_flip.MAX_ABLATIONS_KEY] = 1
    cfs = self.ablation_flip.generate(
        ex, self.classification_model, None, self.classification_config)
    cf_tokens = self.ablation_flip.tokenize(list(cfs)[0]['sentence'])
    self.assertLen(cf_tokens, len(ex_tokens) - 1)

    ex = {'sentence': 'this long movie is terrible and horrible.'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[ablation_flip.MAX_ABLATIONS_KEY] = 1
    cfs = self.ablation_flip.generate(
        ex, self.classification_model, None, self.classification_config)
    self.assertEmpty(cfs)

  def test_ablation_flip_max_ablations_multi_input(self):
    ex = {'sentence1': 'this movie is terrible',
          'sentence2': 'this movie is great'}
    ex_tokens1 = self.ablation_flip.tokenize(ex['sentence1'])
    ex_tokens2 = self.ablation_flip.tokenize(ex['sentence2'])

    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 20
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    max_ablations = 1
    self.regression_config[ablation_flip.MAX_ABLATIONS_KEY] = max_ablations
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    for cf in cfs:
      # Number of ablations in each field should be no more than MAX_ABLATIONS.
      cf_tokens1 = self.ablation_flip.tokenize(cf['sentence1'])
      cf_tokens2 = self.ablation_flip.tokenize(cf['sentence2'])
      self.assertGreaterEqual(len(cf_tokens1), len(ex_tokens1) - max_ablations)
      self.assertGreaterEqual(len(cf_tokens2), len(ex_tokens2) - max_ablations)

  def test_ablation_flip_only_ablate_from_one_field(self):
    ex = {'sentence1': 'this long movie is terrible',
          'sentence2': 'this short movie is great'}
    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 10
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    for cf in cfs:
      self.assertTrue((cf['sentence1'] == ex['sentence1']) or
                      (cf['sentence2'] == ex['sentence2']))

  def test_ablation_flip_changes_pred_class(self):
    ex = {'sentence': 'this long movie is terrible'}
    ex_output = list(self.classification_model.predict([ex]))[0]
    pred_class = str(np.argmax(ex_output['probas']))
    self.assertEqual('0', pred_class)
    cfs = self.ablation_flip.generate(ex, self.classification_model, None,
                                      self.classification_config)
    cf_outputs = self.classification_model.predict(cfs)
    for cf_output in cf_outputs:
      self.assertNotEqual(np.argmax(ex_output['probas']),
                          np.argmax(cf_output['probas']))

  def test_ablation_flip_changes_regression_score(self):
    ex = {'sentence1': 'this long movie is terrible',
          'sentence2': 'this short movie is great'}
    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 2
    ex_output = list(self.regression_model.predict([ex]))[0]
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    cf_outputs = self.regression_model.predict(cfs)
    for cf_output in cf_outputs:
      self.assertNotEqual((ex_output['score'] <= thresh),
                          (cf_output['score'] <= thresh))

  def test_ablation_flip_fails_without_pred_key(self):
    ex = {'sentence': 'this long movie is terrible'}
    with self.assertRaises(AssertionError):
      self.ablation_flip.generate(ex, self.classification_model, None, None)


if __name__ == '__main__':
  absltest.main()
