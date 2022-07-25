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
"""Tests for lit_nlp.components.metrics."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import types
from lit_nlp.components import metrics
from lit_nlp.lib import testing_utils

LitType = types.LitType
DUMMY_MODEL = testing_utils.TestModelBatched()


class RegressionMetricsTest(parameterized.TestCase):

  def test_is_compatible(self):
    """Always false to prevent use as explainer."""
    regression_metrics = metrics.RegressionMetrics()
    self.assertFalse(regression_metrics.is_compatible(DUMMY_MODEL))
    self.assertFalse(regression_metrics.is_compatible(None))

  @parameterized.named_parameters(
      ('regression', types.RegressionScore(), None, True),
      ('mulitclass', types.MulticlassPreds(vocab=['']), None, False),
      ('generated text', types.GeneratedText(), None, False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    regression_metrics = metrics.RegressionMetrics()
    self.assertEqual(
        regression_metrics.is_field_compatible(pred, parent), expected)

  def test_compute_correct(self):
    regression_metrics = metrics.RegressionMetrics()

    result = regression_metrics.compute([1, 2, 3, 4], [1, 2, 3, 4],
                                        types.RegressionScore(),
                                        types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, {
        'mse': 0,
        'pearsonr': 1.0,
        'spearmanr': 1.0
    })

  def test_compute_some_incorrect(self):
    regression_metrics = metrics.RegressionMetrics()

    result = regression_metrics.compute([1, 2, 3, 4], [1, 2, 5.5, 6.3],
                                        types.RegressionScore(),
                                        types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, {
        'mse': 2.885,
        'pearsonr': 0.96566,
        'spearmanr': 1.0
    })

  def test_compute_all_incorrect(self):
    regression_metrics = metrics.RegressionMetrics()

    # All incorrect predictions (and not monotonic).
    result = regression_metrics.compute([1, 2, 3, 4], [-5, -10, 5, 6],
                                        types.RegressionScore(),
                                        types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, {
        'mse': 47.0,
        'pearsonr': 0.79559,
        'spearmanr': 0.799999
    })

  def test_compute_empty_labels(self):
    regression_metrics = metrics.RegressionMetrics()

    result = regression_metrics.compute([], [], types.RegressionScore(),
                                        types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, {})


class MulticlassMetricsTest(parameterized.TestCase):

  def test_is_compatible(self):
    """Always false to prevent use as explainer."""
    multiclass_metrics = metrics.MulticlassMetrics()
    self.assertFalse(multiclass_metrics.is_compatible(DUMMY_MODEL))
    self.assertFalse(multiclass_metrics.is_compatible(None))

  @parameterized.named_parameters(
      ('multiclass', types.MulticlassPreds(vocab=['']), None, True),
      ('regression', types.RegressionScore(), None, False),
      ('generated text', types.GeneratedText(), None, False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    multiclass_metrics = metrics.MulticlassMetrics()
    self.assertEqual(
        multiclass_metrics.is_field_compatible(pred, parent), expected)

  def test_compute_correct(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(
        ['1', '2', '0', '1'], [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1', '2'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {
        'accuracy': 1.0,
        'f1': 1.0,
        'precision': 1.0,
        'recall': 1.0
    })

  def test_compute_some_incorrect(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(
        ['1', '2', '0', '1'],
        [[.1, .4, .5], [0, .1, .9], [.1, 0, .9], [0, 1, 0]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1', '2'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {
        'accuracy': 0.5,
        'f1': 0.57143,
        'precision': 0.5,
        'recall': 0.66667
    })

  def test_compute_all_incorrect(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()
    # All incorrect predictions.
    result = multiclass_metrics.compute(
        ['1', '2', '0', '1'],
        [[.1, .4, .5], [.2, .7, .1], [.1, 0, .9], [1, 0, 0]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1', '2'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {
        'accuracy': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    })

  def test_compute_no_null_index(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(
        ['1', '2', '0', '1'],
        [[.1, .4, .5], [0, .1, .9], [.1, 0, .9], [0, 1, 0]],
        types.CategoryLabel(), types.MulticlassPreds(vocab=['0', '1', '2']))
    testing_utils.assert_deep_almost_equal(self, result, {'accuracy': 0.5})

  def test_compute_correct_single_class(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(['1', '1'], [[.1, .9], [.2, .8]],
                                        types.CategoryLabel(),
                                        types.MulticlassPreds(
                                            vocab=['0', '1'], null_idx=0))
    testing_utils.assert_deep_almost_equal(
        self,
        result,
        {
            'accuracy': 1.0,
            # No AUC in this case.
            'aucpr': 1.0,
            'f1': 1.0,
            'precision': 1.0,
            'recall': 1.0
        })

  def test_compute_almost_correct_single_class_with_null_idx_0(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(['1', '0', '1'],
                                        [[.1, .9], [.9, .1], [.8, .2]],
                                        types.CategoryLabel(),
                                        types.MulticlassPreds(
                                            vocab=['0', '1'], null_idx=0))
    testing_utils.assert_deep_almost_equal(
        self, result, {
            'accuracy': 0.66667,
            'auc': 1.0,
            'aucpr': 1.0,
            'f1': 0.66667,
            'precision': 1.0,
            'recall': 0.5
        })

  def test_compute_almost_correct_multiclass(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute(
        ['1', '0', '2', '3'],
        [[.1, .4, .2, .3], [.9, .1, 0, 0], [0, .3, .5, .2], [.1, .1, .5, .3]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1', '2', '3'], null_idx=0))

    testing_utils.assert_deep_almost_equal(self, result, {
        'accuracy': 0.75,
        'f1': 0.66667,
        'precision': 0.66667,
        'recall': 0.66667,
    })

  def test_compute_empty_labels(self):
    multiclass_metrics = metrics.MulticlassMetricsImpl()

    result = multiclass_metrics.compute([], [], types.CategoryLabel(),
                                        types.MulticlassPreds(
                                            vocab=['0', '1', '2'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {})


class MulticlassPairedMetricsTest(parameterized.TestCase):

  def test_is_compatible(self):
    """Always false to prevent use as explainer."""
    multiclass_paired_metrics = metrics.MulticlassPairedMetrics()
    self.assertFalse(multiclass_paired_metrics.is_compatible(DUMMY_MODEL))
    self.assertFalse(multiclass_paired_metrics.is_compatible(None))

  @parameterized.named_parameters(
      ('multiclass', types.MulticlassPreds(vocab=['']), None, True),
      ('regression', types.RegressionScore(), None, False),
      ('generated text', types.GeneratedText(), None, False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    multiclass_paired_metrics = metrics.MulticlassPairedMetrics()
    self.assertEqual(
        multiclass_paired_metrics.is_field_compatible(pred, parent), expected)

  def test_compute(self):
    multiclass_paired_metrics = metrics.MulticlassPairedMetricsImpl()

    indices = ['7f7f85', '345ac4', '3a3112', '88bcda']
    metas = [{'parentId': '345ac4'}, {}, {}, {'parentId': '3a3112'}]

    # No swaps.
    result = multiclass_paired_metrics.compute_with_metadata(
        ['1', '1', '0', '0'], [[0, 1], [0, 1], [1, 0], [1, 0]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0), indices, metas)
    testing_utils.assert_deep_almost_equal(self, result, {
        'mean_jsd': 0.0,
        'num_pairs': 2,
        'swap_rate': 0.0
    })

    # One swap.
    result = multiclass_paired_metrics.compute_with_metadata(
        ['1', '1', '0', '0'], [[0, 1], [1, 0], [1, 0], [1, 0]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0), indices, metas)
    testing_utils.assert_deep_almost_equal(self, result, {
        'mean_jsd': 0.34657,
        'num_pairs': 2,
        'swap_rate': 0.5
    })

    # Two swaps.
    result = multiclass_paired_metrics.compute_with_metadata(
        ['1', '1', '0', '0'], [[0, 1], [1, 0], [1, 0], [0, 1]],
        types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0), indices, metas)
    testing_utils.assert_deep_almost_equal(self, result, {
        'mean_jsd': 0.69315,
        'num_pairs': 2,
        'swap_rate': 1.0
    })

    # Two swaps, no null index.
    result = multiclass_paired_metrics.compute_with_metadata(
        ['1', '1', '0', '0'], [[0, 1], [1, 0], [1, 0], [0, 1]],
        types.CategoryLabel(), types.MulticlassPreds(vocab=['0', '1']), indices,
        metas)
    testing_utils.assert_deep_almost_equal(self, result, {
        'mean_jsd': 0.69315,
        'num_pairs': 2,
        'swap_rate': 1.0
    })

    # Empty predictions, indices, and meta.
    result = multiclass_paired_metrics.compute_with_metadata(
        [], [], types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0), [], [])
    testing_utils.assert_deep_almost_equal(self, result, {})


class CorpusBLEUTest(parameterized.TestCase):

  def test_is_compatible(self):
    """Always false to prevent use as explainer."""
    bleu_metrics = metrics.CorpusBLEU()
    self.assertFalse(bleu_metrics.is_compatible(DUMMY_MODEL))
    self.assertFalse(bleu_metrics.is_compatible(None))

  @parameterized.named_parameters(
      ('generated text + str', types.GeneratedText(), types.StringLitType(),
       True),
      ('candidates + str', types.GeneratedTextCandidates(),
       types.StringLitType(), True),
      ('bad pred, good parent', types.Scalar(), types.StringLitType(), False),
      ('good pred, bad parent', types.GeneratedText(), types.Scalar(), False),
      ('both bad', types.Scalar(), types.Scalar(), False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    bleu_metrics = metrics.CorpusBLEU()
    self.assertEqual(bleu_metrics.is_field_compatible(pred, parent), expected)

  def test_compute_correct(self):
    bleu_metrics = metrics.CorpusBLEU()

    result = bleu_metrics.compute(
        ['This is a test.', 'Test two', 'A third test example'],
        ['This is a test.', 'Test two', 'A third test example'],
        types.GeneratedText(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result,
                                           {'corpus_bleu': 100.0000})

  def test_compute_some_incorrect(self):
    bleu_metrics = metrics.CorpusBLEU()

    result = bleu_metrics.compute(
        ['This is a test.', 'Test one', 'A third test'],
        ['This is a test.', 'Test two', 'A third test example'],
        types.GeneratedText(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result,
                                           {'corpus_bleu': 68.037493})

    result = bleu_metrics.compute(
        ['This is a test.', 'Test one', 'A third test'],
        ['these test.', 'Test two', 'A third test example'],
        types.GeneratedText(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result,
                                           {'corpus_bleu': 29.508062})

  def test_compute_empty_labels(self):
    bleu_metrics = metrics.CorpusBLEU()

    result = bleu_metrics.compute([], [], types.GeneratedText(),
                                  types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {})

  def test_compute_with_candidates(self):
    bleu_metrics = metrics.CorpusBLEU()

    # Should only score the first one (@1).
    labels = ['This is a test.', 'Test two']
    preds = [
        [('This is a test.', -1.0), ('foobar', -20.0)],
        [('Test two', -1.0), ('spam', -20.0)],
    ]

    result = bleu_metrics.compute(labels, preds, types.TextSegment(),
                                  types.GeneratedTextCandidates())
    testing_utils.assert_deep_almost_equal(self, result,
                                           {'corpus_bleu@1': 100.0000})


class RougeLTest(parameterized.TestCase):

  def test_is_compatible(self):
    """Always false to prevent use as explainer."""
    rouge_metrics = metrics.RougeL()
    self.assertFalse(rouge_metrics.is_compatible(DUMMY_MODEL))
    self.assertFalse(rouge_metrics.is_compatible(None))

  @parameterized.named_parameters(
      ('generated text + str', types.GeneratedText(), types.StringLitType(),
       True),
      ('candidates + str', types.GeneratedTextCandidates(),
       types.StringLitType(), True),
      ('bad pred, good parent', types.Scalar(), types.StringLitType(), False),
      ('good pred, bad parent', types.GeneratedText(), types.Scalar(), False),
      ('both bad', types.Scalar(), types.Scalar(), False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    rouge_metrics = metrics.RougeL()
    self.assertEqual(rouge_metrics.is_field_compatible(pred, parent), expected)

  def test_compute(self):
    rouge_metrics = metrics.RougeL()

    # All correct predictions.
    result = rouge_metrics.compute(
        ['This is a test.', 'Test two', 'A third test example'],
        ['This is a test.', 'Test two', 'A third test example'],
        types.TextSegment(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {'rougeL': 1.0})

    # Some incorrect predictions.
    result = rouge_metrics.compute(
        ['This is a test.', 'Test one', 'A third test'],
        ['This is a test.', 'Test two', 'A third test example'],
        types.TextSegment(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {'rougeL': 0.785714})

    result = rouge_metrics.compute(
        ['This is a test.', 'Test one', 'A third test'],
        ['these test.', 'Test two', 'A third test example'],
        types.TextSegment(), types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {'rougeL': 0.563492})

    # Empty labels and predictions
    result = rouge_metrics.compute([], [], types.GeneratedText(),
                                   types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {})

  def test_compute_with_candidates(self):
    rouge_metrics = metrics.RougeL()

    # Should only score the first one (@1).
    labels = ['This is a test.', 'Test two']
    preds = [
        [('This is a test.', -1.0), ('foobar', -20.0)],
        [('Test two', -1.0), ('spam', -20.0)],
    ]

    result = rouge_metrics.compute(labels, preds, types.TextSegment(),
                                   types.GeneratedTextCandidates())
    testing_utils.assert_deep_almost_equal(self, result, {'rougeL@1': 1.0})


if __name__ == '__main__':
  absltest.main()
