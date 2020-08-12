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
from typing import cast, Dict, List, Tuple, Text, Optional, Sequence, Callable, Any

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
Spec = types.Spec


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


def get_classifications(preds: Sequence[np.ndarray],
                        pred_spec: types.LitType,
                        margin_config: Optional[Text] = None) -> Sequence[int]:
  """Get classified indices given prediction scores and configs."""
  # If there is a margin set for the prediction, take the log of the prediction
  # scores and add the margin to the null indexes value before taking argmax
  # to find the predicted class.
  if margin_config is not None:
    multiclass_pred_spec = cast(types.MulticlassPreds, pred_spec)
    null_idx = multiclass_pred_spec.null_idx
    margin = float(margin_config)
    logit_mask = margin * np.eye(len(multiclass_pred_spec.vocab))[null_idx]
    pred_idxs = [np.argmax(np.log(p) + logit_mask) for p in preds]
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

  def run_with_metadata(self,
                        indexed_inputs: List[JsonDict],
                        model: lit_model.Model,
                        dataset: lit_dataset.Dataset,
                        model_outputs: Optional[List[JsonDict]] = None,
                        config: Optional[JsonDict] = None) -> List[JsonDict]:
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
      metas = [ex['meta'] for ex in indexed_inputs]
      # Compute metrics, as dict(str -> float)
      metrics = self.compute_with_metadata(
          labels,
          preds,
          label_spec=dataset.spec()[label_key],
          pred_spec=spec.output[pred_key],
          indices=indices,
          metas=metas,
          config=config.get(label_key) if config else None)
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
    pearsonr = scipy_stats.pearsonr(labels, preds)[0]
    spearmanr = scipy_stats.spearmanr(labels, preds)[0]
    return {'mse': mse, 'pearsonr': pearsonr, 'spearmanr': spearmanr}


class MulticlassMetrics(SimpleMetrics):
  """Aggregate metrics for multi-class output."""

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


class MulticlassPairedMetrics(SimpleMetrics):
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
      pred_spec: types.LitType,
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


class CorpusBLEU(SimpleMetrics):
  """Corpus BLEU score using SacreBLEU."""

  def is_compatible(self, field_spec: types.LitType) -> bool:
    """Return true if compatible with this field."""
    return isinstance(field_spec, types.GeneratedText)

  def compute(self,
              labels: Sequence[Text],
              preds: Sequence[Text],
              label_spec: types.TextSegment,
              pred_spec: types.GeneratedText,
              config: Optional[JsonDict] = None) -> Dict[Text, float]:
    """Compute metric(s) between labels and predictions."""
    del label_spec
    del pred_spec
    del config

    if not labels or not preds:
      return {}

    bleu = sacrebleu.raw_corpus_bleu(preds, [labels])
    return {'corpus_bleu': bleu.score}
