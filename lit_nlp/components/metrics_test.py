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

from typing import Optional, Union
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import metrics
from lit_nlp.lib import testing_utils

LitType = types.LitType


class TestGenTextModel(lit_model.Model):

  def input_spec(self) -> types.Spec:
    return {'input': types.TextSegment()}

  def output_spec(self) -> types.Spec:
    return {'output': types.GeneratedText(parent='input')}

  def predict_minibatch(self,
                        inputs: list[types.JsonDict]) -> list[types.JsonDict]:
    return [{'output': 'test_output'}] * len(inputs)


class TestGenTextCandsModel(lit_model.Model):

  def input_spec(self) -> types.Spec:
    return {
        'input': types.TextSegment(),
        'label': types.MultiSegmentAnnotations(),
    }

  def output_spec(self) -> types.Spec:
    return {'output': types.GeneratedTextCandidates(parent='input')}

  def predict_minibatch(self,
                        inputs: list[types.JsonDict]) -> list[types.JsonDict]:
    return [
        {'output': [('gen_text one', 0.8), ('gen_text two', 0.3)]}
    ] * len(inputs)


_CLASSIFICATION_MODEL = testing_utils.TestModelClassification()
_GENERATED_TEXT_MODEL = TestGenTextModel()
_GEN_TEXT_CANDS_MODEL = TestGenTextCandsModel()
_REGRESSION_MODEL = testing_utils.TestIdentityRegressionModel()


class RegressionMetricsTest(parameterized.TestCase):

  def setUp(self):
    super(RegressionMetricsTest, self).setUp()
    self.metrics = metrics.RegressionMetrics()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 3)
    self.assertIn('mse', meta_spec)
    self.assertIn('pearsonr', meta_spec)
    self.assertIn('spearmanr', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      ('cls_model', _CLASSIFICATION_MODEL, False),
      ('gen_text_model', _GENERATED_TEXT_MODEL, False),
      ('reg_model', _REGRESSION_MODEL, True),
  )
  def test_is_compatible(self, model: lit_model.Model, expected: bool):
    """Always false to prevent use as explainer."""
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

  @parameterized.named_parameters(
      ('regression', types.RegressionScore(), True),
      ('mulitclass', types.MulticlassPreds(vocab=['']), False),
      ('generated text', types.GeneratedText(), False))
  def test_is_field_compatible(self, pred: LitType, expected: bool):
    self.assertEqual(self.metrics.is_field_compatible(pred, None), expected)

  @parameterized.named_parameters(
      ('correct', [1, 2, 3, 4], [1, 2, 3, 4], 0, 1.0, 1.0),
      ('incorrect', [1, 2, 3, 4], [-5, -10, 5, 6], 47.0, 0.79559, 0.799999),
      ('some_correct', [1, 2, 3, 4], [1, 2, 5.5, 6.3], 2.885, 0.96566, 1.0),
  )
  def test_compute(self, labels: list[float], preds: list[float], mse: float,
                   pearsonr: float, spearmanr: float):
    expected = {'mse': mse, 'pearsonr': pearsonr, 'spearmanr': spearmanr}
    result = self.metrics.compute(labels, preds,
                                  types.RegressionScore(),
                                  types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, expected)

  def test_compute_empty(self):
    result = self.metrics.compute([], [], types.RegressionScore(),
                                  types.RegressionScore())
    testing_utils.assert_deep_almost_equal(self, result, {})


class MulticlassMetricsTest(parameterized.TestCase):

  def setUp(self):
    super(MulticlassMetricsTest, self).setUp()
    self.metrics = metrics.MulticlassMetricsImpl()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 7)
    self.assertIn('accuracy', meta_spec)
    self.assertIn('precision', meta_spec)
    self.assertIn('recall', meta_spec)
    self.assertIn('f1', meta_spec)
    self.assertIn('auc', meta_spec)
    self.assertIn('aucpr', meta_spec)
    self.assertIn('num_missing_labels', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      ('cls_model', _CLASSIFICATION_MODEL, True),
      ('reg_model', _REGRESSION_MODEL, False),
      ('gen_text_model', _GENERATED_TEXT_MODEL, False),
  )
  def test_is_compatible(self, model: lit_model.Model, expected: bool):
    """Always false to prevent use as explainer."""
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

  @parameterized.named_parameters(
      ('multiclass', types.MulticlassPreds(vocab=['']), None, True),
      ('regression', types.RegressionScore(), None, False),
      ('generated text', types.GeneratedText(), None, False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    self.assertEqual(
        self.metrics.is_field_compatible(pred, parent), expected)

  @parameterized.named_parameters(
      (
          'correct', ['0', '1', '2'], ['1', '2', '0', '1'],
          [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
          1.0, 1.0, 1.0, 1.0
      ),
      (
          'incorrect', ['0', '1', '2'], ['1', '2', '0', '1'],
          [[.1, .4, .5], [.2, .7, .1], [.1, 0, .9], [1, 0, 0]],
          0.0, 0.0, 0.0, 0.0
      ),
      (
          'some_correct', ['0', '1', '2'], ['1', '2', '0', '1'],
          [[.1, .4, .5], [0, .1, .9], [.1, 0, .9], [0, 1, 0]],
          0.5, 0.57143, 0.5, 0.66667
      ),
      (
          'some_correct_4_class', ['0', '1', '2', '3'], ['1', '0', '2', '3'],
          [[.1, .4, .2, .3], [.9, .1, 0, 0], [0, .3, .5, .2], [.1, .1, .5, .3]],
          0.75, 0.66667, 0.66667, 0.66667
      ),
  )
  def test_compute_multiclass(
      self, vocab: list[str], labels: list[str], preds: list[list[int]],
      accuracy: float, f1: float, precision: float, recall: float):
    expected = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    result = self.metrics.compute(
        labels, preds, types.CategoryLabel(),
        types.MulticlassPreds(vocab=vocab, null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, expected)

  def test_compute_no_null_index(self):
    result = self.metrics.compute(
        ['1', '2', '0', '1'],
        [[.1, .4, .5], [0, .1, .9], [.1, 0, .9], [0, 1, 0]],
        types.CategoryLabel(), types.MulticlassPreds(vocab=['0', '1', '2']))
    testing_utils.assert_deep_almost_equal(self, result, {'accuracy': 0.5})

  def test_compute_correct_single_class(self):
    result = self.metrics.compute(
        ['1', '1'], [[.1, .9], [.2, .8]], types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {
        'accuracy': 1.0,
        # No AUC in this case.
        'aucpr': 1.0,
        'f1': 1.0,
        'precision': 1.0,
        'recall': 1.0,
    })

  def test_compute_almost_correct_single_class_with_null_idx_0(self):
    result = self.metrics.compute(
        ['1', '0', '1'], [[.1, .9], [.9, .1], [.8, .2]], types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0))
    testing_utils.assert_deep_almost_equal(
        self, result, {
            'accuracy': 0.66667,
            'auc': 1.0,
            'aucpr': 1.0,
            'f1': 0.66667,
            'precision': 1.0,
            'recall': 0.5,
        })

  def test_compute_empty_labels(self):
    result = self.metrics.compute(
        [], [], types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1', '2'], null_idx=0))
    testing_utils.assert_deep_almost_equal(self, result, {})


class MulticlassPairedMetricsTest(parameterized.TestCase):

  def setUp(self):
    super(MulticlassPairedMetricsTest, self).setUp()
    self.metrics = metrics.MulticlassPairedMetricsImpl()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 3)
    self.assertIn('num_pairs', meta_spec)
    self.assertIn('swap_rate', meta_spec)
    self.assertIn('mean_jsd', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      ('cls_model', _CLASSIFICATION_MODEL, True),
      ('reg_model', _REGRESSION_MODEL, False),
      ('gen_text_model', _GENERATED_TEXT_MODEL, False),
  )
  def test_is_compatible(self, model: lit_model.Model, expected: bool):
    """Always false to prevent use as explainer."""
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

  @parameterized.named_parameters(
      ('multiclass', types.MulticlassPreds(vocab=['']), True),
      ('regression', types.RegressionScore(), False),
      ('generated text', types.GeneratedText(), False))
  def test_is_field_compatible(self, pred: LitType, expected: bool):
    self.assertEqual(self.metrics.is_field_compatible(pred, None), expected)

  @parameterized.named_parameters(
      ('no_swaps', [[0, 1], [0, 1], [1, 0], [1, 0]], 0, 0.0, 0.0),
      ('one_swap', [[0, 1], [1, 0], [1, 0], [1, 0]], 0, 0.34657, 0.5),
      ('two_swaps', [[0, 1], [1, 0], [1, 0], [0, 1]], 0, 0.69315, 1.0),
      ('no_null_index', [[0, 1], [1, 0], [1, 0], [0, 1]], None, 0.69315, 1.0),
  )
  def test_compute_with_metadata(self, preds: list[list[int]],
                                 null_idx: Optional[int], mean_jsd: float,
                                 swap_rate: float):

    labels = ['1', '1', '0', '0']
    indices = ['7f7f85', '345ac4', '3a3112', '88bcda']
    metas = [{'parentId': '345ac4'}, {}, {}, {'parentId': '3a3112'}]
    expected = {'mean_jsd': mean_jsd, 'num_pairs': 2, 'swap_rate': swap_rate}
    result = self.metrics.compute_with_metadata(
        labels, preds, types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=null_idx), indices,
        metas)
    testing_utils.assert_deep_almost_equal(self, result, expected)

  def test_compute_with_metadata_empty(self):
    result = self.metrics.compute_with_metadata(
        [], [], types.CategoryLabel(),
        types.MulticlassPreds(vocab=['0', '1'], null_idx=0), [], [])
    testing_utils.assert_deep_almost_equal(self, result, {})


class CorpusBLEUTest(parameterized.TestCase):

  def setUp(self):
    super(CorpusBLEUTest, self).setUp()
    self.metrics = metrics.CorpusBLEU()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 2)
    self.assertIn('corpus_bleu', meta_spec)
    self.assertIn('corpus_bleu@1', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      ('cls_model', _CLASSIFICATION_MODEL, False),
      ('reg_model', _REGRESSION_MODEL, False),
      ('gen_text_model', _GENERATED_TEXT_MODEL, True),
  )
  def test_is_compatible(self, model: lit_model.Model, expected: bool):
    """Always false to prevent use as explainer."""
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

  @parameterized.named_parameters(
      ('generated text, str', types.GeneratedText(), types.StringLitType(),
       True),
      ('candidates, str', types.GeneratedTextCandidates(),
       types.StringLitType(), True),
      ('bad pred, good parent', types.Scalar(), types.StringLitType(), False),
      ('good pred, bad parent', types.GeneratedText(), types.Scalar(), False),
      ('both bad', types.Scalar(), types.Scalar(), False))
  def test_is_field_compatible(self, pred: LitType, parent: LitType,
                               expected: bool):
    self.assertEqual(self.metrics.is_field_compatible(pred, parent), expected)

  @parameterized.named_parameters(
      ('correct', ['This is a test.', 'Test one', 'A third test'], 100.0000),
      (
          'some_different',
          ['This is a test.', 'Test two', 'A third test example'], 68.037493
      ),
      (
          'all_different',
          ['these test.', 'Test two', 'A third test example'], 29.508062
      ),
  )
  def test_compute(self, preds: list[str], score: float):
    labels = ['This is a test.', 'Test one', 'A third test']
    expected = {'corpus_bleu': score}
    result = self.metrics.compute(labels, preds, types.GeneratedText(),
                                  types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, expected)

  def test_compute_empty_labels(self):
    result = self.metrics.compute([], [], types.GeneratedText(),
                                  types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {})

  def test_compute_with_candidates(self):
    # Should only score the first one (@1).
    labels = ['This is a test.', 'Test two']
    preds = [
        [('This is a test.', -1.0), ('foobar', -20.0)],
        [('Test two', -1.0), ('spam', -20.0)],
    ]

    result = self.metrics.compute(labels, preds, types.TextSegment(),
                                  types.GeneratedTextCandidates())
    testing_utils.assert_deep_almost_equal(self, result,
                                           {'corpus_bleu@1': 100.0000})


class RougeLTest(parameterized.TestCase):

  def setUp(self):
    super(RougeLTest, self).setUp()
    self.metrics = metrics.RougeL()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 2)
    self.assertIn('rougeL', meta_spec)
    self.assertIn('rougeL@1', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      ('cls_model', _CLASSIFICATION_MODEL, False),
      ('reg_model', _REGRESSION_MODEL, False),
      ('gen_text_model', _GENERATED_TEXT_MODEL, True),
  )
  def test_is_compatible(self, model: lit_model.Model, expected: bool):
    """Always false to prevent use as explainer."""
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

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
    self.assertEqual(self.metrics.is_field_compatible(pred, parent), expected)

  @parameterized.named_parameters(
      ('correct', ['This is a test.', 'Test one', 'A third test'], 1.0),
      (
          'some_different',
          ['This is a test.', 'Test two', 'A third test example'], 0.785714
      ),
      (
          'all_different',
          ['these test.', 'Test two', 'A third test example'], 0.563492
      ),
  )
  def test_compute(self, preds: list[str], score: float):
    labels = ['This is a test.', 'Test one', 'A third test']
    expected = {'rougeL': score}
    result = self.metrics.compute(labels, preds, types.TextSegment(),
                                  types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, expected)

  def test_compute_empty(self):
    result = self.metrics.compute([], [], types.GeneratedText(),
                                  types.GeneratedText())
    testing_utils.assert_deep_almost_equal(self, result, {})

  def test_compute_with_candidates(self):

    # Should only score the first one (@1).
    labels = ['This is a test.', 'Test two']
    preds = [
        [('This is a test.', -1.0), ('foobar', -20.0)],
        [('Test two', -1.0), ('spam', -20.0)],
    ]

    result = self.metrics.compute(labels, preds, types.TextSegment(),
                                  types.GeneratedTextCandidates())
    testing_utils.assert_deep_almost_equal(self, result, {'rougeL@1': 1.0})


_MULTI_SEG_ANNOTATION_LABELS = [
    [dtypes.AnnotationCluster(label='one', spans=[])],
    [dtypes.AnnotationCluster(label='two', spans=[])],
]


class ExactMatchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.metrics = metrics.ExactMatchMetrics()

  def test_meta_spec(self):
    meta_spec = self.metrics.meta_spec()
    self.assertLen(meta_spec, 2)
    self.assertIn('exactmatch', meta_spec)
    self.assertIn('exactmatch@1', meta_spec)
    for spec in meta_spec.values():
      self.assertIsInstance(spec, types.MetricResult)

  @parameterized.named_parameters(
      dict(
          testcase_name='classification',
          model=_CLASSIFICATION_MODEL,
          expected=False,
      ),
      dict(
          testcase_name='regression',
          model=_REGRESSION_MODEL,
          expected=False,
      ),
      dict(
          testcase_name='gen_text',
          model=_GENERATED_TEXT_MODEL,
          expected=True,
      ),
      dict(
          testcase_name='gen_text_cands',
          model=_GEN_TEXT_CANDS_MODEL,
          expected=True,
      ),
  )
  def test_is_compatible(self, model: LitType, expected: bool):
    compat = self.metrics.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(compat, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='gentext_multi_segment_annotations',
          pred=types.GeneratedText(),
          parent=types.MultiSegmentAnnotations(),
          expected=True,
      ),
      dict(
          testcase_name='gentext_text',
          pred=types.GeneratedText(),
          parent=types.TextSegment(),
          expected=True,
      ),
      dict(
          testcase_name='gencands_multi_segment_annotations',
          pred=types.GeneratedTextCandidates(),
          parent=types.MultiSegmentAnnotations(),
          expected=True,
      ),
      dict(
          testcase_name='gencands_text',
          pred=types.GeneratedTextCandidates(),
          parent=types.TextSegment(),
          expected=True,
      ),
      dict(
          testcase_name='gentext_scalar',
          pred=types.GeneratedText(),
          parent=types.Scalar(),
          expected=False,
      ),
      dict(
          testcase_name='gencands_scalar',
          pred=types.GeneratedTextCandidates(),
          parent=types.Scalar(),
          expected=False,
      ),
      dict(
          testcase_name='text_text',
          pred=types.TextSegment(),
          parent=types.TextSegment(),
          expected=False,
      ),
      dict(
          testcase_name='text_scalar',
          pred=types.TextSegment(),
          parent=types.Scalar(),
          expected=False,
      ),
  )
  def test_is_field_compatible(self,
                               pred: LitType,
                               parent: LitType,
                               expected: bool):
    self.assertEqual(self.metrics.is_field_compatible(pred, parent), expected)

  @parameterized.named_parameters(
      # Without labels or preds, it should return an empty dict
      dict(
          testcase_name='no_labels',
          labels=[],
          preds=['one', 'two'],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedText(),
          expected={},
      ),
      dict(
          testcase_name='no_preds',
          labels=['one', 'two'],
          preds=[],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedText(),
          expected={},
      ),
      # Tests for all, some, and none correct w/ MultiSegmentAnnotations labels
      dict(
          testcase_name='correct_multi_segment_annotations_gentext',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=['one', 'two'],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 1.0},
      ),
      dict(
          testcase_name='correct_multi_segment_annotations_gencands',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=[[('one', None)], [('two', None)]],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 1.0},
      ),
      dict(
          testcase_name='some_multi_segment_annotations_gentext',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=['one', 'four'],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 0.5},
      ),
      dict(
          testcase_name='some_multi_segment_annotations_gencands',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=[[('one', None)], [('four', None)]],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 0.5},
      ),
      dict(
          testcase_name='none_multi_segment_annotations_gentext',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=['three', 'four'],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 0.0},
      ),
      dict(
          testcase_name='none_multi_segment_annotations_gencands',
          labels=_MULTI_SEG_ANNOTATION_LABELS,
          preds=[[('three', None)], [('four', None)]],
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 0.0},
      ),
      # Tests for all, some, and none correct w/ TextSegment labels
      dict(
          testcase_name='correct_text_gentext',
          labels=['one', 'two'],
          preds=['one', 'two'],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 1.0},
      ),
      dict(
          testcase_name='correct_text_gencands',
          labels=['one', 'two'],
          preds=[[('one', None)], [('two', None)]],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 1.0},
      ),
      dict(
          testcase_name='some_text_gentext',
          labels=['one', 'two'],
          preds=['one', 'four'],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 0.5},
      ),
      dict(
          testcase_name='some_text_gencands',
          labels=['one', 'two'],
          preds=[[('one', None)], [('four', None)]],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 0.5},
      ),
      dict(
          testcase_name='none_text_gentext',
          labels=['one', 'two'],
          preds=['three', 'four'],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedText(),
          expected={'exactmatch': 0.0},
      ),
      dict(
          testcase_name='none_text_gencands',
          labels=['one', 'two'],
          preds=[[('three', None)], [('four', None)]],
          label_spec=types.TextSegment(),
          preds_spec=types.GeneratedTextCandidates(),
          expected={'exactmatch@1': 0.0},
      ),
  )
  def test_compute(self,
                   labels: Union[list[str],
                                 list[list[dtypes.AnnotationCluster]]],
                   preds,
                   label_spec: Union[types.MultiSegmentAnnotations,
                                     types.TextSegment],
                   preds_spec: Union[types.GeneratedText,
                                     types.GeneratedTextCandidates],
                   expected: dict[str, float]):
    result = self.metrics.compute(labels, preds, label_spec, preds_spec)
    testing_utils.assert_deep_almost_equal(self, result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid_labels_gentext',
          label_spec=types.Scalar(),
          preds_spec=types.GeneratedText(),
      ),
      dict(
          testcase_name='invalid_labels_gentextcandidates',
          label_spec=types.Scalar(),
          preds_spec=types.GeneratedTextCandidates(),
      ),
      dict(
          testcase_name='invalid_preds_text',
          label_spec=types.TextSegment(),
          preds_spec=types.Scalar(),
      ),
      dict(
          testcase_name='invalid_preds_multi_segment_annotations',
          label_spec=types.MultiSegmentAnnotations(),
          preds_spec=types.Scalar(),
      ),
  )
  def test_compute_spec_exceptions(self,
                                   label_spec: types.LitType,
                                   preds_spec: types.LitType):
    inputs = ['one', 'two', 'three']
    preds = ['one', 'two', 'three']
    with self.assertRaises(TypeError):
      self.metrics.compute(inputs, preds, label_spec, preds_spec)


if __name__ == '__main__':
  absltest.main()
