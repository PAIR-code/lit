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

from typing import Iterable, Iterator

from absl.testing import absltest
from lit_nlp.api import types
from lit_nlp.components import ablation_flip
from lit_nlp.examples.models import glue_models
import numpy as np

# TODO(lit-dev): Move glue_models out of lit_nlp/examples


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
STSB_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/stsb_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
  extract_compressed_file=True)
STSB_PATH = transformers.file_utils.cached_path(STSB_PATH,
  extract_compressed_file=True)


class SST2ModelNonRequiredField(glue_models.SST2Model):

  def input_spec(self):
    spec = super().input_spec()
    spec['sentence'] = types.TextSegment(required=False, default='')
    return spec


class SST2ModelWithPredictCounter(glue_models.SST2Model):

  def __init__(self, *args, **kw):
    super().__init__(*args, **kw)
    self.predict_counter = 0

  def predict(self,
              inputs: Iterable[types.JsonDict],
              scrub_arrays=True,
              **kw) -> Iterator[types.JsonDict]:
    results = super().predict(inputs, scrub_arrays, **kw)
    self.predict_counter += 1
    return results


class ModelBasedAblationFlipTest(absltest.TestCase):

  def setUp(self):
    super(ModelBasedAblationFlipTest, self).setUp()
    self.ablation_flip = ablation_flip.AblationFlip()

    # Classification model that clasifies a given input sentence.
    self.classification_model = glue_models.SST2Model(BERT_TINY_PATH)
    self.classification_config = {ablation_flip.PREDICTION_KEY: 'probas'}

    # Clasification model with the 'sentence' field marked as
    # non-required.
    self.classification_model_non_required_field = SST2ModelNonRequiredField(
        BERT_TINY_PATH)

    # Clasification model with a counter to count number of predict calls.
    # TODO(ataly): Consider setting up a Mock object to count number of
    # predict calls.
    self.classification_model_with_predict_counter = (
        SST2ModelWithPredictCounter(BERT_TINY_PATH))

    # Regression model determining similarity between two input sentences.
    self.regression_model = glue_models.STSBModel(STSB_PATH)
    self.regression_config = {ablation_flip.PREDICTION_KEY: 'score'}

  def test_ablation_flip_num_ex(self):
    ex = {'sentence': 'this long movie was terrible'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 0
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
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
    self.regression_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence1',
        'sentence2',
    ]
    self.assertLen(
        self.ablation_flip.generate(ex, self.regression_model, None,
                                    self.regression_config), 2)

  def test_ablation_flip_long_sentence(self):
    sentence = (
        'this was a terrible terrible movie but I am a writing '
        'a nice long review for testing whether AblationFlip '
        'can handle long sentences with a bounded number of '
        'predict calls.')
    ex = {'sentence': sentence}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 100
    self.classification_config[ablation_flip.MAX_ABLATIONS_KEY] = 100
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
    model = self.classification_model_with_predict_counter
    cfs = self.ablation_flip.generate(
        ex, model, None, self.classification_config)

    # This example must yield 19 ablation_flips.
    self.assertLen(cfs, 19)

    # Number of predict calls made by ablation_flip should be upper-bounded by
    # <number of tokens in sentence> + 2**MAX_ABLATABLE_TOKENS
    num_tokens = len(model.tokenizer(sentence))
    num_predict_calls = model.predict_counter
    self.assertLessEqual(num_predict_calls,
                         num_tokens + 2**ablation_flip.MAX_ABLATABLE_TOKENS)

    # We use a smaller value of MAX_ABLATABLE_TOKENS and check that the
    # number of predict calls is smaller, and that the prediction bound still
    # holds.
    model.predict_counter = 0
    ablation_flip.MAX_ABLATABLE_TOKENS = 5
    self.assertLessEqual(model.predict_counter, num_predict_calls)
    self.assertLessEqual(model.predict_counter,
                         num_tokens + 2**ablation_flip.MAX_ABLATABLE_TOKENS)

  def test_ablation_flip_freeze_fields(self):
    ex = {'sentence1': 'this long movie is terrible',
          'sentence2': 'this long movie is great'}
    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 10
    thresh = 2
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = thresh
    self.regression_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence1'
    ]
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    for cf in cfs:
      self.assertEqual(cf['sentence2'], ex['sentence2'])

  def test_ablation_flip_max_ablations(self):
    ex = {'sentence': 'this movie is terrible'}
    ex_tokens = self.ablation_flip.tokenize(ex['sentence'])
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[ablation_flip.MAX_ABLATIONS_KEY] = 1
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
    cfs = self.ablation_flip.generate(
        ex, self.classification_model, None, self.classification_config)
    cf_tokens = self.ablation_flip.tokenize(list(cfs)[0]['sentence'])
    self.assertLen(cf_tokens, len(ex_tokens) - 1)

    ex = {'sentence': 'this long movie is terrible and horrible.'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[ablation_flip.MAX_ABLATIONS_KEY] = 1
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
    cfs = self.ablation_flip.generate(
        ex, self.classification_model, None, self.classification_config)
    self.assertEmpty(cfs)

  def test_ablation_flip_max_ablations_multi_input(self):
    ex = {'sentence1': 'this movie is terrible',
          'sentence2': 'this movie is great'}
    ex_tokens1 = self.ablation_flip.tokenize(ex['sentence1'])
    ex_tokens2 = self.ablation_flip.tokenize(ex['sentence2'])

    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 20
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = 2
    max_ablations = 1
    self.regression_config[ablation_flip.MAX_ABLATIONS_KEY] = max_ablations
    self.regression_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence1',
        'sentence2',
    ]
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)
    for cf in cfs:
      # Number of ablations in each field should be no more than MAX_ABLATIONS.
      cf_tokens1 = self.ablation_flip.tokenize(cf['sentence1'])
      cf_tokens2 = self.ablation_flip.tokenize(cf['sentence2'])
      self.assertGreaterEqual(
          len(cf_tokens1) + len(cf_tokens2),
          len(ex_tokens1) + len(ex_tokens2) - max_ablations)

  def test_ablation_flip_yields_multi_field_ablations(self):
    ex = {'sentence1': 'this short movie is awesome',
          'sentence2': 'this short movie is great'}
    ex_tokens1 = self.ablation_flip.tokenize(ex['sentence1'])
    ex_tokens2 = self.ablation_flip.tokenize(ex['sentence2'])

    self.regression_config[ablation_flip.NUM_EXAMPLES_KEY] = 20
    self.regression_config[ablation_flip.REGRESSION_THRESH_KEY] = 2
    self.regression_config[ablation_flip.MAX_ABLATIONS_KEY] = 5
    self.regression_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence1',
        'sentence2',
    ]
    cfs = self.ablation_flip.generate(ex, self.regression_model, None,
                                      self.regression_config)

    # Verify that at least one counterfactual involves ablations across
    # multiple fields.
    multi_field_ablation_found = False
    for cf in cfs:
      cf_tokens1 = self.ablation_flip.tokenize(cf['sentence1'])
      cf_tokens2 = self.ablation_flip.tokenize(cf['sentence2'])
      if ((len(cf_tokens1) < len(ex_tokens1))
          and (len(cf_tokens2) < len(ex_tokens2))):
        multi_field_ablation_found = True
        break
    self.assertTrue(multi_field_ablation_found)

  def test_ablation_flip_changes_pred_class(self):
    ex = {'sentence': 'this long movie is terrible'}
    ex_output = list(self.classification_model.predict([ex]))[0]
    pred_class = str(np.argmax(ex_output['probas']))
    self.assertEqual('0', pred_class)
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
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
    self.regression_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence1',
        'sentence2',
    ]
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

  def test_ablation_flip_required_field(self):
    ex = {'sentence': 'terrible'}
    self.classification_config[ablation_flip.NUM_EXAMPLES_KEY] = 1
    self.classification_config[ablation_flip.FIELDS_TO_ABLATE_KEY] = [
        'sentence'
    ]
    self.assertEmpty(
        self.ablation_flip.generate(
            ex, self.classification_model, None, self.classification_config))
    self.assertLen(
        self.ablation_flip.generate(
            ex, self.classification_model_non_required_field,
            None, self.classification_config), 1)

if __name__ == '__main__':
  absltest.main()
