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
"""Metric component and implementations."""

import abc
import collections
import numbers
from typing import cast, Dict, List, Sequence, Tuple, Text, Optional, Callable, Any, Union

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np
import sacrebleu
from scipy import stats as scipy_stats
from scipy.spatial import distance as scipy_distance
from sklearn import metrics as sklearn_metrics

JsonDict = types.JsonDict
IndexedInput = types.IndexedInput
Spec = types.Spec

BLEU_SMOOTHING_VAL = 0.1


def map_pred_keys(
    data_spec: lit_model.Spec, model_output_spec: lit_model.Spec,
    predicate: Callable[[types.LitType], bool]) -> Dict[Text, Text]:
  """Find output fields matching predicate, and return a mapping to input fields."""
  ret = {}
  for pred_key in utils.find_keys(model_output_spec, predicate):
    pred_field_spec = model_output_spec[pred_key]
    label_key = getattr(pred_field_spec, 'parent', None)
    if label_key is None:
      logging.warning("Pred key '%s' has no parent field. Skipping.", pred_key)
      continue  # skip fields with no pointer
    if label_key not in data_spec:
      # This may be intentional, if running on unlabeled data.
      logging.warning(
          "Pred key '%s' points to missing label field '%s'. Skipping.",
          pred_key, label_key)
      continue
    ret[pred_key] = label_key
  return ret


def get_margin_for_input(margin_config: Optional[JsonDict] = None,
                         inp: Optional[JsonDict] = None):
  """Get margin given a margin config and input example."""
  if not margin_config:
    return 0

  for margin_entry in margin_config.values():
    facet_info = (margin_entry['facetData']['facets']
                  if 'facetData' in margin_entry else {})
    match = True
    if inp:
      for feat, facet_info in facet_info.items():
        value = facet_info['val']
        if (isinstance(inp[feat], numbers.Number) and
            not isinstance(inp[feat], bool)):
          # If the facet is a numeric range string, extract the min and max
          # and check the value against that range.
          min_val = value[0]
          max_val = value[1]
          if not (inp[feat] >= min_val and inp[feat] < max_val):
            match = False
        # If the facet is a standard value, check the feature value for
        # equality to it.
        elif inp[feat] != value:
          match = False
    if match:
      return margin_entry['margin']
  return 0


def get_classifications(
    preds: Sequence[np.ndarray], pred_spec: types.MulticlassPreds,
    margin_config: Optional[Sequence[float]] = None) -> Sequence[int]:
  """Get classified indices given prediction scores and configs."""
  # If there is a margin set for the prediction, take the log of the prediction
  # scores and add the margin to the null indexes value before taking argmax
  # to find the predicted class.
  if margin_config is not None:
    multiclass_pred_spec = cast(types.MulticlassPreds, pred_spec)
    null_idx = multiclass_pred_spec.null_idx
    pred_idxs = []
    for p, margin in zip(preds, margin_config):
      logit_mask = margin * np.eye(len(multiclass_pred_spec.vocab))[null_idx]
      pred_idx = np.argmax(np.log(p) + logit_mask)
      pred_idxs.append(pred_idx)
  else:
    pred_idxs = [np.argmax(p) for p in preds]
  return pred_idxs


class SimpleMetrics(lit_components.Interpreter):
  """Base class for simple metrics, which should render in the main metrics table."""

  @abc.abstractmethod
  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    pass

  def compute(self,
              labels: Sequence[Any],
              preds: Sequence[Any],
              label_spec: types.LitType,
              pred_spec: types.LitType,
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    raise NotImplementedError(
        'Subclass should implement this, or override compute_with_metadata() directly.'
    )

  def compute_with_metadata(
      self,
      labels: Sequence[Any],
      preds: Sequence[Any],
      label_spec: types.LitType,
      pred_spec: types.LitType,
      indices: Sequence[types.ExampleId],
      metas: Sequence[JsonDict],
      config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """As compute(), but has access to indices and metadata."""
    return self.compute(labels, preds, label_spec, pred_spec, config)

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    spec = model.spec()
    field_map = map_pred_keys(dataset.spec(), spec.output, self.is_compatible)
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
      # NaN is not a valid JSON value, so replace with None which will be
      # serialized as null.
      # TODO(lit-team): move this logic into serialize.py somewhere instead?
      metrics = {
          k: (v if not np.isnan(v) else None) for k, v in metrics.items()
      }
      # Format for frontend.
      ret.append({
          'pred_key': pred_key,
          'label_key': label_key,
          'metrics': metrics
      })
    return ret

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None) -> List[JsonDict]:
    if model_outputs is None:
      model_outputs = list(model.predict_with_metadata(indexed_inputs))

    # TODO(lit-team): pre-compute this mapping in constructor?
    # This would require passing a model name to this function so we can
    # reference a pre-computed list.
    spec = model.spec()
    field_map = map_pred_keys(dataset.spec(), spec.output, self.is_compatible)
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
      # NaN is not a valid JSON value, so replace with None which will be
      # serialized as null.
      # TODO(lit-team): move this logic into serialize.py somewhere instead?
      metrics = {
          k: (v if not np.isnan(v) else None) for k, v in metrics.items()
      }
      # Format for frontend.
      ret.append({
          'pred_key': pred_key,
          'label_key': label_key,
          'metrics': metrics
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

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return self._metrics.is_compatible(field_spec)

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None):
    # Get margin for each input for each pred key and add them to a config dict
    # to pass to the wrapped metrics.
    field_map = map_pred_keys(dataset.spec(),
                              model.spec().output, self.is_compatible)
    margin_config = {}
    for pred_key in field_map:
      field_config = config.get(pred_key) if config else None
      margins = [get_margin_for_input(field_config, inp) for inp in inputs]
      margin_config[pred_key] = margins
    return self._metrics.run(inputs, model, dataset, model_outputs,
                             margin_config)

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None) -> List[JsonDict]:
    # Get margin for each input for each pred key and add them to a config dict
    # to pass to the wrapped metrics.
    field_map = map_pred_keys(
        dataset.spec(), model.spec().output, self.is_compatible)
    margin_config = {}
    for pred_key in field_map:
      inputs = [ex['data'] for ex in indexed_inputs]
      field_config = config.get(pred_key) if config else None
      margins = [get_margin_for_input(field_config, inp) for inp in inputs]
      margin_config[pred_key] = margins
    return self._metrics.run_with_metadata(
        indexed_inputs, model, dataset, model_outputs, margin_config)


class RegressionMetrics(SimpleMetrics):
  """Standard regression metrics."""

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec, types.RegressionScore)

  def compute(self,
              labels: Sequence[float],
              preds: Sequence[float],
              label_spec: types.Scalar,
              pred_spec: types.RegressionScore,
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    del config

    if not labels or not preds:
      return {}

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
                      y_pred: Sequence[int],
                      vocab: Sequence[Text],
                      null_idx: Optional[int] = None):
    # Filter out unlabeled examples before calculating metrics.
    total_len = len(y_true)
    labeled_example_indices = [
        index for index, y in enumerate(y_true) if y != -1
    ]
    y_true = [y_true[i] for i in labeled_example_indices]
    y_pred = [y_pred[i] for i in labeled_example_indices]

    ret = collections.OrderedDict()
    ret['accuracy'] = sklearn_metrics.accuracy_score(y_true, y_pred)
    # TODO(lit-team): compute macro averages as well?

    # If task has a null class then compute P,R,F1 by treating
    # null_idx as the negative / "other" class.
    if null_idx is not None:
      # Note: labels here are indices.
      labels: List[int] = [i for i in range(len(vocab)) if i != null_idx]
      ret['precision'] = sklearn_metrics.precision_score(
          y_true, y_pred, labels=labels, average='micro')
      ret['recall'] = sklearn_metrics.recall_score(
          y_true, y_pred, labels=labels, average='micro')
      ret['f1'] = sklearn_metrics.f1_score(
          y_true, y_pred, labels=labels, average='micro')

    if len(labeled_example_indices) != total_len:
      ret['num_missing_labels'] = total_len - len(labeled_example_indices)

    return ret

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec, types.MulticlassPreds)

  def compute(self,
              labels: Sequence[Text],
              preds: Sequence[np.ndarray],
              label_spec: types.CategoryLabel,
              pred_spec: types.MulticlassPreds,
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    # TODO(lit-dev): compare on strings instead of converting to indices?
    # This should be more robust to skew in label sets.
    del label_spec  # Unused; get vocab from pred_spec.

    if not labels or not preds:
      return {}

    label_idxs = [
        pred_spec.vocab.index(label) if label in pred_spec.vocab else -1
        for label in labels
    ]
    pred_idxs = get_classifications(preds, pred_spec, config)
    return self.get_all_metrics(
        label_idxs, pred_idxs, pred_spec.vocab, null_idx=pred_spec.null_idx)


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

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec, types.MulticlassPreds)

  @staticmethod
  def find_pairs(indices: Sequence[types.ExampleId],
                 metas: Sequence[JsonDict]) -> List[Tuple[int, int]]:
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
      label_spec: types.LitType,
      pred_spec: types.MulticlassPreds,
      indices: Sequence[types.ExampleId],
      metas: Sequence[JsonDict],
      config: Optional[JsonDict] = None) -> Dict[Text, float]:
    del labels  # Unused; we only care about preds.
    del label_spec  # Unused; we only care about preds.

    ret = collections.OrderedDict()

    pairs = self.find_pairs(indices, metas)
    ret['num_pairs'] = len(pairs)
    if ret['num_pairs'] == 0:
      return {}

    pred_idxs = get_classifications(preds, pred_spec, config)

    # 'swapped' just means the prediction changed.
    is_swapped = [(pred_idxs[i] == pred_idxs[j]) for i, j in pairs]
    ret['swap_rate'] = 1 - np.mean(is_swapped)

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

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec,
                      (types.GeneratedText, types.GeneratedTextCandidates))

  def compute(self,
              labels: Sequence[Text],
              preds: Sequence[Union[Text, types.ScoredTextCandidates]],
              label_spec: types.TextSegment,
              pred_spec: Union[types.GeneratedText,
                               types.GeneratedTextCandidates],
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    del label_spec
    del config

    if not labels or not preds:
      return {}

    name_suffix = ''
    if isinstance(pred_spec, types.GeneratedTextCandidates):
      preds = [types.GeneratedTextCandidates.top_text(v) for v in preds]
      name_suffix = '@1'
    bleu = sacrebleu.raw_corpus_bleu(preds, [labels], BLEU_SMOOTHING_VAL)

    return {'corpus_bleu' + name_suffix: bleu.score}


class BinaryConfusionMetricsImpl(SimpleMetrics):
  """Confusion matrix values for binary classification."""

  def get_all_metrics(self,
                      y_true: Sequence[int],
                      y_pred: Sequence[int],
                      vocab: Sequence[Text],
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

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if binary classification with ground truth."""
    if not isinstance(field_spec, types.MulticlassPreds):
      return False
    class_spec = cast(types.MulticlassPreds, field_spec)
    return len(class_spec.vocab) == 2 and class_spec.parent

  def compute(self,
              labels: Sequence[Text],
              preds: Sequence[np.ndarray],
              label_spec: types.CategoryLabel,
              pred_spec: types.MulticlassPreds,
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    del label_spec  # Unused; get vocab from pred_spec.

    if not labels or not preds:
      return {}

    label_idxs = [
        pred_spec.vocab.index(label) if label in pred_spec.vocab else -1
        for label in labels
    ]
    # Get classifications using possible margin value to control threshold
    # of positive classification.
    pred_idxs = get_classifications(preds, pred_spec, config)

    return self.get_all_metrics(
        label_idxs, pred_idxs, pred_spec.vocab, null_idx=pred_spec.null_idx)


class BinaryConfusionMetrics(ClassificationMetricsWrapper):

  def __init__(self):
    ClassificationMetricsWrapper.__init__(self, BinaryConfusionMetricsImpl())
