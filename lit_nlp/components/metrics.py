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
"""Metric component and implementations."""

import collections
from typing import Any, Callable, Optional, Sequence, Union, cast

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import classification_results
import numpy as np
import sacrebleu
from scipy import stats as scipy_stats
from scipy.spatial import distance as scipy_distance
from sklearn import metrics as sklearn_metrics

from rouge_score import rouge_scorer

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
LitType = types.LitType
Spec = types.Spec


def map_pred_keys(
    data_spec: Spec, model_output_spec: Spec,
    predicate: Callable[[LitType, Optional[LitType]], bool]) -> dict[str, str]:
  """Returns a map of compatible output fields and their parent input fields."""
  ret = {}
  for pred_key, pred_spec in model_output_spec.items():
    parent_key: Optional[str] = getattr(pred_spec, 'parent', None)
    if parent_key is None:
      logging.info("Skipping '%s': No parent provided.", pred_key)
      continue

    parent_spec: Optional[LitType] = data_spec.get(parent_key)
    if predicate(pred_spec, parent_spec):
      ret[pred_key] = parent_key
    else:
      logging.info("Skipping '%s': incompatible parent '%s'.", pred_key,
                   parent_key)
      continue
  return ret


def nan_to_none(metrics: dict[str, float]) -> dict[str, Optional[float]]:
  # NaN is not a valid JSON value, so replace with None which will be
  # serialized as null.
  # TODO(lit-dev): consider moving this logic to serialize.py?
  return {k: (v if not np.isnan(v) else None) for k, v in metrics.items()}


class SimpleMetrics(lit_components.Metrics):
  """Base class for built-in metrics rendered in the main metrics table."""

  def run(self,
          inputs: Sequence[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    spec = model.spec()
    field_map = map_pred_keys(dataset.spec(), spec.output,
                              self.is_field_compatible)
    ret = []
    for pred_key, label_key in field_map.items():
      # Extract fields
      labels = [ex[label_key] for ex in inputs]
      preds = [mo[pred_key] for mo in model_outputs]
      # Compute metrics, as dict(str -> float)
      metrics = self.compute(
          labels,
          preds,
          label_spec=dataset.spec()[label_key],
          pred_spec=spec.output[pred_key],
          config=config.get(pred_key) if config else None)
      # Format for frontend.
      ret.append({
          'pred_key': pred_key,
          'label_key': label_key,
          'metrics': nan_to_none(metrics)
      })
    return ret

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[list[JsonDict]] = None,
                        config: Optional[JsonDict] = None) -> list[JsonDict]:
    if model_outputs is None:
      model_outputs = list(model.predict_with_metadata(indexed_inputs))

    # TODO(lit-team): pre-compute this mapping in constructor?
    # This would require passing a model name to this function so we can
    # reference a pre-computed list.
    spec = model.spec()
    field_map = map_pred_keys(dataset.spec(), spec.output,
                              self.is_field_compatible)
    ret = []
    for pred_key, label_key in field_map.items():
      # Extract fields
      labels = [ex['data'][label_key] for ex in indexed_inputs]
      preds = [mo[pred_key] for mo in model_outputs]
      indices = [ex['id'] for ex in indexed_inputs]
      metas = [ex.get('meta', {}) for ex in indexed_inputs]
      # Compute metrics, as dict(str -> float)
      metrics = self.compute_with_metadata(
          labels,
          preds,
          label_spec=dataset.spec()[label_key],
          pred_spec=spec.output[pred_key],
          indices=indices,
          metas=metas,
          config=config.get(pred_key) if config else None)
      # Format for frontend.
      ret.append({
          'pred_key': pred_key,
          'label_key': label_key,
          'metrics': nan_to_none(metrics)
      })
    return ret


class ClassificationMetricsWrapper(lit_components.Interpreter):
  """Wrapper for classification metrics interpreters.

  Gets margin setting for each input based on raw scores and on provided
  margins in the config, which can be faceted by input feature values. Then
  passes the raw scores example-specific margins to the metrics interpreter that
  this class wraps for calculation of metrics.
  """

  def __init__(self, metrics: SimpleMetrics):
    self._metrics = metrics

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    """Metrics should always return false for Model-level compatibility."""
    return self._metrics.is_compatible(model, dataset)

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    return self._metrics.is_field_compatible(pred_spec, parent_spec)

  def meta_spec(self) -> dict[str, types.LitType]:
    return self._metrics.meta_spec()

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    # Get margin for each input for each pred key and add them to a config dict
    # to pass to the wrapped metrics.
    field_map = map_pred_keys(dataset.spec(),
                              model.spec().output, self.is_field_compatible)
    margin_config = {}
    for pred_key in field_map:
      field_config = config.get(pred_key) if config else None
      margins = [
          classification_results.get_margin_for_input(field_config, inp)
          for inp in inputs
      ]
      margin_config[pred_key] = margins
    return self._metrics.run(inputs, model, dataset, model_outputs,
                             margin_config)

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[list[JsonDict]] = None,
                        config: Optional[JsonDict] = None) -> list[JsonDict]:
    # Get margin for each input for each pred key and add them to a config dict
    # to pass to the wrapped metrics.
    field_map = map_pred_keys(dataset.spec(),
                              model.spec().output, self.is_field_compatible)
    margin_config = {}
    for pred_key in field_map:
      inputs = [ex['data'] for ex in indexed_inputs]
      field_config = config.get(pred_key) if config else None
      margins = [
          classification_results.get_margin_for_input(field_config, inp)
          for inp in inputs
      ]
      margin_config[pred_key] = margins
    return self._metrics.run_with_metadata(indexed_inputs, model, dataset,
                                           model_outputs, margin_config)


class RegressionMetrics(SimpleMetrics):
  """Standard regression metrics."""

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    del parent_spec
    return isinstance(pred_spec, types.RegressionScore)

  def meta_spec(self) -> dict[str, types.LitType]:
    return {
        'mse': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='Mean squared error: Estimates the mean of the '
                        'square of the differences between the estimated value '
                        'and the actual value. Closer to 0 is better.'),
        'pearsonr': types.MetricResult(
            description="Pearson's R: Measures the linear correlation between "
                        "the estimated value and the actual value. Values "
                        "closer to 1 indicate a strong positive correlation "
                        "and values closee to -1 indicate a strong negative "
                        "correlation."),
        'spearmanr': types.MetricResult(
            description="Spearman's Rho: Measures the rank correlation between "
                        "the estimated and actual values. Values closer to 1 "
                        "indicate a strong positive correlation and values "
                        "closer to -1 indicate a strong negative correlation."),
    }

  def compute(self,
              labels: Sequence[float],
              preds: Sequence[float],
              label_spec: LitType,
              pred_spec: LitType,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute the MSE and Pearson's & Spearman's R for regression predictions.

    Args:
      labels: Ground truth values for each prediction.
      preds: The models predicted regression scores, aligned with `labels`.
      label_spec: A Scalar spec for the model's label field.
      pred_spec: A RegressionScore spec for the model's prediction field.
      config: Unused configuration dict inherited from super class.

    Returns:
      A dict containing the mean squared error (key=`mse`), Pearson's R
      (key=`pearsonr`) and Spearmean's R (key=`spearmanr`) for each prediction.
      If `preds` or `labels` are empty, returns an empty dict.

    Raises:
      TypeError: `label_spec` is not a `Scalar` or `pred_spec` is not a
        `RegressionScore`. Note overriding the type information in the method
        signature will produce a signature mismatch error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    del config

    if not labels or not preds:
      return {}

    if not isinstance(label_spec, types.Scalar):
      raise TypeError('label_spec must be a Scalar, received '
                      f'{type(label_spec).__name__}')

    if not isinstance(pred_spec, types.RegressionScore):
      raise TypeError('pred_spec must be a RegressionScore, received '
                      f'{type(pred_spec).__name__}')

    mse = sklearn_metrics.mean_squared_error(labels, preds)
    if len(labels) < 2:  # Check if only one point selected.
      pearsonr = np.nan
    else:
      pearsonr = scipy_stats.pearsonr(labels, preds)[0]
    spearmanr = scipy_stats.spearmanr(labels, preds)[0]
    return {'mse': mse, 'pearsonr': pearsonr, 'spearmanr': spearmanr}


class MulticlassMetricsImpl(SimpleMetrics):
  """Aggregate metrics for multi-class output."""

  def get_all_metrics(self,
                      y_true: Sequence[int],
                      y_pred_probs: Sequence[np.ndarray],
                      pred_spec: types.MulticlassPreds,
                      config: Optional[JsonDict] = None,
                      null_idx: Optional[int] = None):

    # Filter out unlabeled examples before calculating metrics.
    total_len = len(y_true)
    labeled_example_indices = [
        index for index, y in enumerate(y_true) if y != -1
    ]
    y_true = [y_true[i] for i in labeled_example_indices]
    y_pred_probs = [y_pred_probs[i] for i in labeled_example_indices]
    y_pred = classification_results.get_classifications(y_pred_probs, pred_spec,
                                                        config)
    y_pred = [y_pred[i] for i in labeled_example_indices]

    ret = collections.OrderedDict()
    ret['accuracy'] = sklearn_metrics.accuracy_score(y_true, y_pred)
    # TODO(lit-team): compute macro averages as well?

    # If task has a null class then compute P,R,F1 by treating
    # null_idx as the negative / "other" class.
    if null_idx is not None:
      # Note: labels here are indices.
      labels: list[int] = [
          i for i in range(len(pred_spec.vocab)) if i != null_idx
      ]
      ret['precision'] = sklearn_metrics.precision_score(
          y_true, y_pred, labels=labels, average='micro')
      ret['recall'] = sklearn_metrics.recall_score(
          y_true, y_pred, labels=labels, average='micro')
      ret['f1'] = sklearn_metrics.f1_score(
          y_true, y_pred, labels=labels, average='micro')

      # The target type used in computing metrics will be 'binary'.
      # Reshape predictions to only include those of the positive class.
      if len(pred_spec.vocab) == 2:
        y_score = [1 - p[null_idx] for p in y_pred_probs
                  ]  # <float[]>[num_examples]

        y_true_indicators = [y != null_idx for y in y_true]
        # AUC is not defined when there is only 1 unique class.
        if len(np.unique(y_true)) > 1:
          ret['auc'] = sklearn_metrics.roc_auc_score(
              y_true_indicators, y_score, average='micro')
        ret['aucpr'] = sklearn_metrics.average_precision_score(
            y_true_indicators, y_score, average='micro')

    if len(labeled_example_indices) != total_len:
      ret['num_missing_labels'] = total_len - len(labeled_example_indices)

    return ret

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    del parent_spec
    return isinstance(pred_spec, types.MulticlassPreds)

  def meta_spec(self) -> dict[str, types.LitType]:
    return {
        'accuracy': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The proportion of correct labels predicted by the '
                        'model. Closer to 1 is better.'),
        'precision': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The proportion of correct predictions for this class '
                        'out of all predictions of this class. Closer to 1 is '
                        'better.'),
        'recall': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The proportion of correct predictions for this class '
                        'out of all datapoints in this class. Closer to 1 is '
                        'better.'),
        'f1': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The performance of the model as the harmonic mean of '
                        'precision and recall. Closer to 1 is better.'),
        'auc': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='Area under the ROC curve. Closer to 1 is better.'),
        'aucpr': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='Area under the PR curve. Closer to 1 is better.'),
        'num_missing_labels': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='The number of predictions that did not have ground '
                        'truth labels. Closer to 0 is better.'),
    }

  def compute(self,
              labels: Sequence[str],
              preds: Sequence[np.ndarray],
              label_spec: LitType,
              pred_spec: LitType,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute standard metrics for multiclass predictions.

    Args:
      labels: Ground truth class for each prediction.
      preds: The models predicted class label, aligned with `labels`.
      label_spec: Unused field spec from super class
      pred_spec: A MulticlassPreds spec for the model's prediction field.
      config: Unused configuration dict inherited from super class.

    Returns:
      A dict containing the `accuracy`, `precission`, `recall`, `f1`, `auc`,
      `aucpr`, and `num_missing_labels` scores for the provided predictions.
      If `preds` or `labels` are empty, returns an empty dict.

    Raises:
      TypeError: `pred_spec` is not `MulticlassPreds`. Note overriding the type
        information in the method signature will produce a signature mismatch
        error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    # TODO(lit-dev): compare on strings instead of converting to indices?
    # This should be more robust to skew in label sets.
    del label_spec  # Unused; get vocab from pred_spec.

    if not labels or not preds:
      return {}

    if not isinstance(pred_spec, types.MulticlassPreds):
      raise TypeError('pred_spec must be a MulticlassPreds, received '
                      f'{type(pred_spec).__name__}')

    label_idxs = [
        pred_spec.vocab.index(label) if label in pred_spec.vocab else -1
        for label in labels
    ]
    return self.get_all_metrics(
        label_idxs,
        preds,
        pred_spec,
        null_idx=pred_spec.null_idx,
        config=config)


class MulticlassMetrics(ClassificationMetricsWrapper):

  def __init__(self):
    ClassificationMetricsWrapper.__init__(self, MulticlassMetricsImpl())


class MulticlassPairedMetricsImpl(SimpleMetrics):
  """Paired analysis between generated datapoints and their parents.

  Currently, this computes the swap rate, which is a measure of how often the
  generated datapoint causes the model to change its prediction. We also report
  mean JSD between model(d) and model(d') as a "soft" measure of the response of
  the model to the perturbations.
  """

  def meta_spec(self) -> types.Spec:
    return {
        'num_pairs': types.MetricResult(
            description='The number of pairs found/analyzed.'),
        'swap_rate': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='The proportion of time the prediction differs between '
                        'the pair of examples. Closer to 0 is better.'),
        'mean_jsd': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='Mean Jensen-Shannon distance measures the similarity '
                        'between two probability distributions. Closer to 0 is '
                        'better.'),
    }

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    del parent_spec
    return isinstance(pred_spec, types.MulticlassPreds)

  @staticmethod
  def find_pairs(indices: Sequence[types.ExampleId],
                 metas: Sequence[JsonDict]) -> list[tuple[int, int]]:
    """Find valid pairs in the current selection, and return list indices."""
    id_to_position = {example_id: i for i, example_id in enumerate(indices)}
    pairs = []  # (i,j) relative to labels and preds lists
    for this_id, meta in zip(indices, metas):
      if 'parentId' not in meta:
        continue  # skip if no parent
      parent_id = meta['parentId']
      if parent_id not in id_to_position:
        continue  # skip if parent not in current selection
      pairs.append((id_to_position[parent_id], id_to_position[this_id]))
    return pairs

  def compute_with_metadata(
      self,
      labels: Sequence[Any],
      preds: Sequence[Any],
      label_spec: LitType,
      pred_spec: LitType,
      indices: Sequence[types.ExampleId],
      metas: Sequence[JsonDict],
      config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute standard paired metrics for multiclass predictions.

    Args:
      labels: Unused list of ground truth values from the super class.
      preds: The models predicted class label, aligned with `labels`.
      label_spec: Unused field spec from the super class.
      pred_spec: A MulticlassPreds spec for the model's prediction field.
      indices: The ID for each IndexedInput, aligned with `preds` and `labels`.
      metas: The metadata for each Input, aligned with `preds` and `labels`.
      config: Optional margins for computing classification results.

    Returns:
      A dict containing the `num_pairs`, `swap_rate`, and `mean_jsd` values for
      the provided `preds`. If `num_pairs` is 0, returns an empty dict.

    Raises:
      TypeError: `pred_spec` is not `MulticlassPreds`. Note overriding the type
        information in the method signature will produce a signature mismatch
        error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    del labels  # Unused; we only care about preds.
    del label_spec  # Unused; we only care about preds.

    ret = collections.OrderedDict()

    pairs = self.find_pairs(indices, metas)
    ret['num_pairs'] = len(pairs)
    if ret['num_pairs'] == 0:
      return {}

    if not isinstance(pred_spec, types.MulticlassPreds):
      raise TypeError('pred_spec must be a MulticlassPreds, received '
                      f'{type(pred_spec).__name__}')

    pred_idxs = classification_results.get_classifications(
        preds, pred_spec, config)

    # 'swapped' just means the prediction changed.
    is_swapped = [(pred_idxs[i] != pred_idxs[j]) for i, j in pairs]
    ret['swap_rate'] = np.mean(is_swapped)

    # Jensen-Shannon divergence, as a soft measure of prediction change.
    jsds = [
        scipy_distance.jensenshannon(preds[i], preds[j])**2 for i, j in pairs
    ]
    ret['mean_jsd'] = np.mean(jsds)

    return ret


class MulticlassPairedMetrics(ClassificationMetricsWrapper):

  def __init__(self):
    ClassificationMetricsWrapper.__init__(self, MulticlassPairedMetricsImpl())


class CorpusBLEU(SimpleMetrics):
  """Corpus BLEU score using SacreBLEU."""

  BLEU_SMOOTHING_VAL = 0.1

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    is_pred_comaptible = isinstance(
        pred_spec, (types.GeneratedText, types.GeneratedTextCandidates))
    is_parent_compatible = isinstance(parent_spec, types.StringLitType)
    return is_pred_comaptible and is_parent_compatible

  def meta_spec(self) -> dict[str, types.LitType]:
    return {
        'corpus_bleu': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='BLEU score, a measure of text quality, over an entire '
                        'corpus. Closer to 1 is better.'),
        'corpus_bleu@1': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='BLEU score, a measure of text quality, over an entire '
                        'corpus for the top predicted candidate. Closer to 1 '
                        'is better.'),
    }

  def compute(self,
              labels: Sequence[str],
              preds: Sequence[Union[str, types.ScoredTextCandidates]],
              label_spec: LitType,
              pred_spec: LitType,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute CorpusBLEU score using the SacreBLEU library.

    Args:
      labels: Ground truth values for each prediction.
      preds: The models predicted values, aligned with `labels`.
      label_spec: Unused field spec from super class.
      pred_spec: A `GeneratedText` or `GeneratedTextCandidates` spec for the
        model's prediction field.
      config: Unused configuration dict inherited from super class.

    Returns:
      A dict containing the CorpusBLEU score for each prediction, stored in the
      `corpus_bleu` key if `pred_spec` is `GeneratedText` or the
      `corpus_bleu@1` key if `pred_spec` is `GeneratedTextCandidates`.
      If `preds` or `labels` are empty, returns an empty dict.

    Raises:
      TypeError: `pred_spec` is not `GeneratedText`/`GeneratedTextCandidates`.
        Note overriding the type information in the method signature will
        produce a signature mismatch error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    del label_spec
    del config

    if not labels or not preds:
      return {}

    if not isinstance(pred_spec,
                      (types.GeneratedText, types.GeneratedTextCandidates)):
      raise TypeError('pred_spec must be a GeneratedText or '
                      'GeneratedTextCandidates, received '
                      f'{type(pred_spec).__name__}')

    name_suffix = ''
    if isinstance(pred_spec, types.GeneratedTextCandidates):
      preds = [types.GeneratedTextCandidates.top_text(v) for v in preds]
      name_suffix = '@1'
    bleu = sacrebleu.raw_corpus_bleu(preds, [labels], self.BLEU_SMOOTHING_VAL)

    return {'corpus_bleu' + name_suffix: bleu.score}


class RougeL(SimpleMetrics):
  """RougeL score for generation tasks."""

  def __init__(self, *args, **kw):
    super().__init__(*args, **kw)
    self._scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  def _score(self, reference, prediction):
    return self._scorer.score(
        target=reference, prediction=prediction)['rougeL'].fmeasure

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field."""
    is_pred_comaptible = isinstance(
        pred_spec, (types.GeneratedText, types.GeneratedTextCandidates))
    is_parent_compatible = isinstance(parent_spec, types.StringLitType)
    return is_pred_comaptible and is_parent_compatible

  def meta_spec(self) -> dict[str, types.LitType]:
    return {
        'rougeL': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='ROUGE score, a measure of text quality, for the '
                        'longest common subsequence in the text. Closer to 1 '
                        'is better.'),
        'rougeL@1': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='ROUGE score, a measure of text quality, for the '
                        'longest common subsequence in the text for the top '
                        'predicted candidate. Closer to 1 is better.')
    }

  def compute(self,
              labels: Sequence[str],
              preds: Sequence[Union[str, types.ScoredTextCandidates]],
              label_spec: LitType,
              pred_spec: LitType,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute the RougeL score using the RougeScorer library.

    Args:
      labels: Ground truth values for each prediction.
      preds: The models predicted values, aligned with `labels`.
      label_spec: Unused field spec from super class.
      pred_spec: A `GeneratedText` or `GeneratedTextCandidates` spec for the
        model's prediction field.
      config: Unused configuration dict inherited from super class.

    Returns:
      A dict containing the RougeL score for each prediction, stored in the
      `rougeL` key if `pred_spec` is `GeneratedText` or the `rougeL@1` key if
      `pred_spec` is `GeneratedTextCandidates`. If `preds` or `labels` are
      empty, returns an empty dict.

    Raises:
      TypeError: `pred_spec` is not `GeneratedText`/`GeneratedTextCandidates`.
        Note overriding the type information in the method signature will
        produce a signature mismatch error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    del label_spec
    del config

    if not labels or not preds:
      return {}

    if not isinstance(pred_spec,
                      (types.GeneratedText, types.GeneratedTextCandidates)):
      raise TypeError('pred_spec must be a GeneratedText or '
                      'GeneratedTextCandidates, received '
                      f'{type(pred_spec).__name__}')

    name_suffix = ''
    if isinstance(pred_spec, types.GeneratedTextCandidates):
      preds = [types.GeneratedTextCandidates.top_text(v) for v in preds]
      name_suffix = '@1'
    scores = list(map(self._score, labels, preds))

    return {'rougeL' + name_suffix: np.mean(scores)}


class BinaryConfusionMetricsImpl(SimpleMetrics):
  """Confusion matrix values for binary classification."""

  def get_all_metrics(self,
                      y_true: Sequence[int],
                      y_pred: Sequence[int],
                      vocab: Sequence[str],
                      null_idx: Optional[int] = None):
    # Filter out unlabeled examples before calculating metrics.
    labeled_example_indices = [
        index for index, y in enumerate(y_true) if y != -1
    ]
    y_true = [y_true[i] for i in labeled_example_indices]
    y_pred = [y_pred[i] for i in labeled_example_indices]

    # Return binary confusion matrix entries.
    ret = collections.OrderedDict()
    matrix = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    ret['TN'] = matrix[0][0]
    ret['FP'] = matrix[0][1]
    ret['FN'] = matrix[1][0]
    ret['TP'] = matrix[1][1]
    return ret

  def meta_spec(self) -> dict[str, types.LitType]:
    return {
        'FN': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='The number of false negatives predicted by the model. '
                        'Closer to 0 is better.'),
        'FP': types.MetricResult(
            best_value=types.MetricBestValue.ZERO,
            description='The number of false positives predicted by the model. '
                        'Closer to 0 is better.'),
        'TN': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The number of true negatives predicted by the model. '
                        'Higher is better.'),
        'TP': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The number of true positives predicted by the model. '
                        'Hugher is better.'),
    }

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if binary classification with ground truth."""
    if not (isinstance(pred_spec, types.MulticlassPreds) and
            isinstance(parent_spec, types.CategoryLabel)):
      return False
    class_spec = cast(types.MulticlassPreds, pred_spec)
    return len(class_spec.vocab) == 2

  def compute(self,
              labels: Sequence[str],
              preds: Sequence[np.ndarray],
              label_spec: LitType,
              pred_spec: LitType,
              config: Optional[JsonDict] = None) -> dict[str, float]:
    """Compute binary classification metrics using Scikit-Learn.

    Args:
      labels: Ground truth class label for each prediction.
      preds: The models predicted class label, aligned with `labels`.
      label_spec: Unused field spec from the super class.
      pred_spec: A `MulticlassPreds` spec for the model's prediction field.
      config: Optional margins for computing classification results.

    Returns:
      A dict containing the true negative (`TN`), false positive (`FP`), false
      negative (`FN`), and true positive (`TN`) scores. If `labels` or `preds`
      is empty, returns an empty dict.

    Raises:
      TypeError: `pred_spec` is not `MulticlassPreds`. Note overriding the type
        information in the method signature will produce a signature mismatch
        error in PyType, see
        https://google.github.io/pytype/errors.html#signature-mismatch
    """
    del label_spec  # Unused; get vocab from pred_spec.

    if not labels or not preds:
      return {}

    if not isinstance(pred_spec, types.MulticlassPreds):
      raise TypeError('pred_spec must be a MulticlassPreds, received '
                      f'{type(pred_spec).__name__}')

    label_idxs = [
        pred_spec.vocab.index(label) if label in pred_spec.vocab else -1
        for label in labels
    ]
    # Get classifications using possible margin value to control threshold
    # of positive classification.
    pred_idxs = classification_results.get_classifications(
        preds, pred_spec, config)

    return self.get_all_metrics(
        label_idxs, pred_idxs, pred_spec.vocab, null_idx=pred_spec.null_idx)


class BinaryConfusionMetrics(ClassificationMetricsWrapper):

  def __init__(self):
    ClassificationMetricsWrapper.__init__(self, BinaryConfusionMetricsImpl())


class ExactMatchMetrics(SimpleMetrics):
  """Exact match metrics for text generations."""

  def meta_spec(self) -> types.Spec:
    """Returns the spec for the Exact Match metrics.

    Returns
      A dict of MetricResult specs for the metrics computed by this class.
    """
    return {
        'exactmatch': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The proportion of exact matches. Closer to 1 is '
                        'better.',
        ),
        'exactmatch@1': types.MetricResult(
            best_value=types.MetricBestValue.HIGHEST,
            description='The proportion of exact matches for the top predicted '
                        'candidate. Closer to 1 is better.',
        )
    }

  def is_field_compatible(self, pred_spec: LitType,
                          parent_spec: Optional[LitType]) -> bool:
    """Return true if compatible with this field.

    Args:
      pred_spec: The field in the model's output spec containing the generated
          text, must be of type GeneratedText or GeneratedTextCandidates.
      parent_spec: The field in the dataset containing the ground truth, must be
          of type MultiSegmentAnnotations or TextSegment.

    Returns:
      True if the pred_spec and parent_spec pair are compatible.
    """
    pred_supported = isinstance(pred_spec, (types.GeneratedText,
                                            types.GeneratedTextCandidates))
    parent_supported = isinstance(parent_spec, (types.TextSegment,
                                                types.MultiSegmentAnnotations))
    return pred_supported and parent_supported

  def compute(
      self,
      labels: Sequence[Any],
      preds: Sequence[Any],
      label_spec: types.LitType,
      pred_spec: types.LitType,
      config: Optional[JsonDict] = None) -> lit_components.MetricsDict:
    """Compute exact matches between labels and predictions.

    Args:
      labels: Ground truth against which predictions are compared.
      preds: The predictions made by the model.
      label_spec: A `MultiSegmentAnnotations` or `TextSegment` spec  describing
          the types of elements in `labels`.
      pred_spec: A `GeneratedText` or `GeneratedTextCandidates` spec describing
          the types of elements in `preds`.
      config: unused parameter from base class.

    Returns:
      A dict containing the proportion of exact matches in the predictions,
      stored in the `exactmatch` key if `pred_spec` is `GeneratedText` or the
      `exactmatch@1` key if `pred_spec` is `GeneratedTextCandidates`.
    """
    del config

    if not labels or not preds:
      return {}

    if not isinstance(label_spec,
                      (types.TextSegment, types.MultiSegmentAnnotations)):
      raise TypeError('label_spec must be a TextSegment or '
                      'MultiSegmentAnnotations, received '
                      f'{type(pred_spec).__name__}')

    if not isinstance(pred_spec,
                      (types.GeneratedText, types.GeneratedTextCandidates)):
      raise TypeError('pred_spec must be a GeneratedText or '
                      'GeneratedTextCandidates, received '
                      f'{type(pred_spec).__name__}')

    if isinstance(pred_spec, types.GeneratedTextCandidates):
      texts = [types.GeneratedTextCandidates.top_text(v) for v in preds]
      name_suffix = '@1'
    else:
      texts = preds
      name_suffix = ''

    matches = 0
    for label, pred in zip(labels, texts):
      if isinstance(label_spec, types.MultiSegmentAnnotations):
        # MultiSegmentAnnotations means that labels is a
        # Sequence[api.dtypes.AnnotationCluster].
        answers = [annotation.label for annotation in label]
        if any(pred == answer for answer in answers):
          matches += 1
      else:
        # Otherwise, labels is a Sequence[str].
        if pred == label:
          matches += 1

    return {f'exactmatch{name_suffix}': matches/len(preds)}
