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
"""Tests for lit_nlp.components.hotflip."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.components import hotflip
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


_CONFIG_CLASSIFICATION = {
    hotflip.FIELDS_TO_HOTFLIP_KEY: ['tokens_sentence'],
    hotflip.PREDICTION_KEY: 'probas',
}
_CONFIG_REGRESSION = {
    hotflip.FIELDS_TO_HOTFLIP_KEY: ['tokens_sentence1', 'tokens_sentence2'],
    hotflip.PREDICTION_KEY: 'score',
    hotflip.REGRESSION_THRESH_KEY: 2,
}

_SST2_EXAMPLE = {'sentence': 'this long movie is terrible.'}
_STSB_EXAMPLE = {
    'sentence1': 'this long movie is terrible.',
    'sentence2': 'this short movie is great.'
}


class HotflipIntegrationTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(HotflipIntegrationTest, self).__init__(*args, **kwargs)
    self.classification_model = glue_models.SST2Model(BERT_TINY_PATH)
    self.regression_model = glue_models.STSBModel(STSB_PATH)

  def setUp(self):
    super(HotflipIntegrationTest, self).setUp()
    self.hotflip = hotflip.HotFlip()

  @parameterized.named_parameters(
      ('0_examples', 0),
      ('1_examples', 1),
      ('2_examples', 2),
  )
  def test_hotflip_num_ex(self, num_examples: int):
    config = {**_CONFIG_CLASSIFICATION, hotflip.NUM_EXAMPLES_KEY: num_examples}
    counterfactuals = self.hotflip.generate(_SST2_EXAMPLE,
                                            self.classification_model, None,
                                            config)
    self.assertLen(counterfactuals, num_examples)

  @parameterized.named_parameters(
      ('0_examples', 0),
      ('1_examples', 1),
      ('2_examples', 2),
  )
  def test_hotflip_num_ex_multi_input(self, num_examples: int):
    config = {**_CONFIG_REGRESSION, hotflip.NUM_EXAMPLES_KEY: num_examples}
    counterfactuals = self.hotflip.generate(_STSB_EXAMPLE,
                                            self.regression_model, None, config)
    self.assertLen(counterfactuals, num_examples)

  @parameterized.named_parameters(
      ('terrible', ['terrible'], [4]),
      ('long_terrible', ['long', 'terrible'], [1, 4]),
  )
  def test_hotflip_freeze_tokens(self, ignore: list[str],
                                 exp_indexes: list[int]):
    config = {
        **_CONFIG_CLASSIFICATION,
        hotflip.NUM_EXAMPLES_KEY: 10,
        hotflip.TOKENS_TO_IGNORE_KEY: ignore,
    }

    counterfactuals = self.hotflip.generate(
        _SST2_EXAMPLE, self.classification_model, None, config)
    self.assertEqual(len(ignore), len(exp_indexes))
    for target, index in zip(ignore, exp_indexes):
      for counterfactual in counterfactuals:
        tokens = counterfactual['tokens_sentence']
        self.assertEqual(target, tokens[index])

  def test_hotflip_freeze_tokens_multi_input(self):
    config = {
        **_CONFIG_REGRESSION,
        hotflip.NUM_EXAMPLES_KEY: 10,
        hotflip.TOKENS_TO_IGNORE_KEY: ['long', 'terrible'],
    }

    ex = {
        'sentence1': 'this long movie is terrible.',
        'sentence2': 'this long movie is great.',
    }

    counterfactuals = self.hotflip.generate(ex, self.regression_model, None,
                                            config)

    for cf in counterfactuals:
      tokens1 = cf['tokens_sentence1']
      tokens2 = cf['tokens_sentence2']
      self.assertEqual('terrible', tokens1[4])
      self.assertEqual('long', tokens1[1])
      self.assertEqual('long', tokens2[1])

  def test_hotflip_max_flips(self):
    config = {
        **_CONFIG_CLASSIFICATION,
        hotflip.MAX_FLIPS_KEY: 1,
        hotflip.NUM_EXAMPLES_KEY: 1,
    }
    ex = _SST2_EXAMPLE

    ex_output = list(self.classification_model.predict([ex]))[0]
    ex_tokens = ex_output['tokens_sentence']
    cfs_1 = self.hotflip.generate(ex, self.classification_model, None, config)
    cf_tokens = list(cfs_1)[0]['tokens_sentence']

    ex = {'sentence': 'this long movie is terrible and horrible.'}
    cfs_2 = self.hotflip.generate(ex, self.classification_model, None, config)

    self.assertEqual(1, sum([1 for i, t in enumerate(cf_tokens)
                             if t != ex_tokens[i]]))
    self.assertEmpty(cfs_2)

  def test_hotflip_max_flips_multi_input(self):
    config = {
        **_CONFIG_REGRESSION,
        hotflip.MAX_FLIPS_KEY: 1,
        hotflip.NUM_EXAMPLES_KEY: 20,
    }
    ex = _STSB_EXAMPLE
    ex_output = list(self.regression_model.predict([ex]))[0]
    ex_tokens1 = ex_output['tokens_sentence1']
    ex_tokens2 = ex_output['tokens_sentence2']
    cfs = self.hotflip.generate(ex, self.regression_model, None, config)
    for cf in cfs:
      # Number of flips in each field should be no more than MAX_FLIPS.
      cf_tokens1 = cf['tokens_sentence1']
      cf_tokens2 = cf['tokens_sentence2']
      self.assertLessEqual(sum([1 for i, t in enumerate(cf_tokens1)
                                if t != ex_tokens1[i]]), 1)
      self.assertLessEqual(sum([1 for i, t in enumerate(cf_tokens2)
                                if t != ex_tokens2[i]]), 1)

  def test_hotflip_only_flip_one_field(self):
    config = {**_CONFIG_REGRESSION, hotflip.NUM_EXAMPLES_KEY: 10}
    ex = _STSB_EXAMPLE
    cfs = self.hotflip.generate(ex, self.regression_model, None, config)
    for cf in cfs:
      self.assertTrue(
          (cf['sentence1'] == ex['sentence1']) or
          (cf['sentence2'] == ex['sentence2']))

  def test_hotflip_changes_pred_class(self):
    config = _CONFIG_CLASSIFICATION
    ex = _SST2_EXAMPLE

    ex_output = list(self.classification_model.predict([ex]))[0]
    pred_class = str(np.argmax(ex_output['probas']))
    cfs = self.hotflip.generate(ex, self.classification_model, None, config)
    cf_outputs = self.classification_model.predict(cfs)

    self.assertEqual('0', pred_class)
    for cf_output in cf_outputs:
      self.assertNotEqual(np.argmax(ex_output['probas']),
                          np.argmax(cf_output['probas']))

  def test_hotflip_changes_regression_score(self):
    config = {**_CONFIG_REGRESSION, hotflip.NUM_EXAMPLES_KEY: 2}
    ex = _STSB_EXAMPLE

    thresh = config[hotflip.REGRESSION_THRESH_KEY]
    ex_output = list(self.regression_model.predict([ex]))[0]
    cfs = self.hotflip.generate(ex, self.regression_model, None, config)
    cf_outputs = self.regression_model.predict(cfs)
    for cf_output in cf_outputs:
      self.assertNotEqual((ex_output['score'] <= thresh),
                          (cf_output['score'] <= thresh))


if __name__ == '__main__':
  absltest.main()
