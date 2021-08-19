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
"""Tests for lit_nlp.components.thresholder."""

from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.components import thresholder
# TODO(lit-dev): Move glue_models out of lit_nlp/examples
from lit_nlp.examples.models import glue_models
from lit_nlp.lib import caching  # for hash id fn


JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


BERT_TINY_PATH = 'https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz'  # pylint: disable=line-too-long
import transformers
BERT_TINY_PATH = transformers.file_utils.cached_path(BERT_TINY_PATH,
extract_compressed_file=True)


class ThresholderTest(absltest.TestCase):

  def setUp(self):
    super(ThresholderTest, self).setUp()
    self.thresholder = thresholder.Thresholder()
    self.model = caching.CachingModelWrapper(
        glue_models.SST2Model(BERT_TINY_PATH), 'test')
    examples = [
        {'sentence': 'a', 'label': '1'},
        {'sentence': 'b', 'label': '1'},
        {'sentence': 'c', 'label': '1'},
        {'sentence': 'd', 'label': '1'},
        {'sentence': 'e', 'label': '1'},
        {'sentence': 'f', 'label': '0'},
        {'sentence': 'g', 'label': '0'},
        {'sentence': 'h', 'label': '0'},
        {'sentence': 'i', 'label': '0'}]

    self.indexed_inputs = [{'id': caching.input_hash(ex), 'data': ex}
                           for ex in examples]
    self.dataset = lit_dataset.IndexedDataset(
        id_fn=caching.input_hash,
        spec={'sentence': lit_types.TextSegment(),
              'label': lit_types.CategoryLabel(vocab=['0', '1'])},
        indexed_examples=self.indexed_inputs)
    self.model_outputs = list(self.model.predict_with_metadata(
        self.indexed_inputs, dataset_name='test'))

  def test_thresholder(self):
    # Test with default options.
    config = None
    result = self.thresholder.run_with_metadata(
        self.indexed_inputs, self.model, self.dataset, self.model_outputs,
        config=config)
    self.assertLen(result, 1)
    self.assertEqual('probas', result[0]['pred_key'])
    self.assertEqual(0.71, result[0]['thresholds']['']['Single'])

  def test_thresholder_cost_ratio_high(self):
    config = {'cost_ratio': 5, 'facets': {'': {}}}
    result = self.thresholder.run_with_metadata(
        self.indexed_inputs, self.model, self.dataset, self.model_outputs,
        config=config)
    self.assertEqual(0.86, result[0]['thresholds']['']['Single'])

  def test_thresholder_cost_ratio_low(self):
    config = {'cost_ratio': 0.2, 'facets': {'': {}}}
    result = self.thresholder.run_with_metadata(
        self.indexed_inputs, self.model, self.dataset, self.model_outputs,
        config=config)
    self.assertEqual(0, result[0]['thresholds']['']['Single'])

  def test_thresholder_facets(self):
    config = {'cost_ratio': 1, 'facets': {
        'label:1': {'data': self.indexed_inputs[0:5]},
        'label:0': {'data': self.indexed_inputs[5:9]}}}
    result = self.thresholder.run_with_metadata(
        self.indexed_inputs, self.model, self.dataset, self.model_outputs,
        config=config)
    print(result)
    self.assertEqual(0.71, result[0]['thresholds']['label:0']['Single'])
    self.assertEqual(0.71, result[0]['thresholds']['label:1']['Single'])
    self.assertEqual(0.86, result[0]['thresholds']['label:0']['Individual'])
    self.assertEqual(0,
                     result[0]['thresholds']['label:0']['Demographic parity'])
    self.assertEqual(0, result[0]['thresholds']['label:1']['Individual'])


if __name__ == '__main__':
  absltest.main()
