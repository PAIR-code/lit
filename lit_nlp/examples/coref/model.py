"""LIT model implementation for frozen-encoder coreference."""
import os
from typing import List

from absl import logging
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.coref import edge_predictor
from lit_nlp.examples.coref import encoders
from lit_nlp.examples.coref.datasets import winogender
from lit_nlp.lib import utils
import numpy as np
import tqdm

EdgeLabel = lit_dtypes.EdgeLabel
JsonDict = lit_types.JsonDict


class FrozenEncoderCoref(lit_model.Model):
  """Frozen-encoder coreference model."""

  @classmethod
  def from_saved(cls, path: str, encoder_cls, classifier_cls):
    """Reload from the output of .save()."""
    encoder_path = os.path.join(path, 'encoder')
    encoder = encoder_cls(encoder_path)
    classifier_path = os.path.join(path, 'classifier')
    classifier = classifier_cls(classifier_path)
    return cls(encoder, classifier)

  def __init__(self, encoder: encoders.BertEncoderWithOffsets,
               classifier: edge_predictor.SingleEdgePredictor):
    self.encoder = encoder
    self.classifier = classifier

    embs_field = utils.find_spec_keys(self.encoder.output_spec(),
                                      lit_types.TokenEmbeddings)[0]
    offset_field = utils.find_spec_keys(self.encoder.output_spec(),
                                        lit_types.SubwordOffsets)[0]
    self.extractor_kw = dict(
        edge_field='coref', embs_field=embs_field, offset_field=offset_field)

  def _make_edges(self, inputs: List[JsonDict], show_progress=False):
    if show_progress:
      progress = lambda x: tqdm.tqdm(x, total=len(inputs))
    else:
      progress = lambda x: x
    return edge_predictor.EdgeFeaturesDataset.build(
        inputs,
        self.encoder,
        progress=progress,
        verbose=show_progress,
        **self.extractor_kw)

  def train(self, train_inputs: List[JsonDict],
            validation_inputs: List[JsonDict], **train_kw):
    # Extract encoder features.
    logging.info('Train: extracting span representations...')
    train_edges = self._make_edges(train_inputs, show_progress=True)
    logging.info('Train: %d edge targets from %d inputs.', len(train_edges),
                 len(train_inputs))
    logging.info('Validation: extracting span representations...')
    validation_edges = self._make_edges(validation_inputs, show_progress=True)
    logging.info('Validation: %d edge targets from %d inputs.',
                 len(validation_edges), len(validation_inputs))

    # Train classifier layer.
    history = self.classifier.train(train_edges.examples,
                                    validation_edges.examples, **train_kw)
    return history

  def save(self, path: str):
    if not os.path.isdir(path):
      os.mkdir(path)
    self.classifier.save(os.path.join(path, 'classifier'))
    self.encoder.save(os.path.join(path, 'encoder'))

  ##
  # LIT API implementations
  def max_minibatch_size(self):
    return self.encoder.max_minibatch_size()

  def predict_minibatch(self, inputs: List[JsonDict]):
    edges = self._make_edges(inputs, show_progress=False)
    edge_preds = list(self.classifier.predict(edges.examples))
    # Re-pack outputs to align with inputs.
    preds = [{'coref': [], 'tokens': ex['tokens']} for ex in inputs]
    for edge, ep in zip(edges.examples, edge_preds):
      orig_input = inputs[edge['src_idx']]
      orig_edge = orig_input['coref'][edge['edge_idx']]
      new_edge = EdgeLabel(
          span1=orig_edge.span1, span2=orig_edge.span2, label=ep['proba'])
      preds[edge['src_idx']]['coref'].append(new_edge)
    for ex, p in zip(inputs, preds):
      # Choose an answer if there are only two target edges.
      if len(p['coref']) == 2 and 'answer' in ex:
        probas = np.array([ep.label for ep in p['coref']])
        # Renormalize as a binary choice.
        p['pred_answer'] = probas / np.sum(probas)
      # Otherwise, it's ok for this field to be missing;
      # metrics will safely ignore.
    return preds

  def input_spec(self):
    return {
        'text':
            lit_types.TextSegment(),
        'tokens':
            lit_types.Tokens(parent='text'),
        'coref':
            lit_types.EdgeLabels(align='tokens'),
        # Index of predicted (single) edge for Winogender
        'answer':
            lit_types.CategoryLabel(
                vocab=winogender.ANSWER_VOCAB, required=False),
        # TODO(b/172975096): allow plotting of scalars from input data,
        # so we don't need to add this to the predictions.
        'pf_bls':
            lit_types.Scalar(required=False),
    }

  def output_spec(self):
    # TODO(lit-dev): also return the embeddings for each span on datasets
    # with a fixed number of targets; for Winogender this would be
    # {occupation, other participant, pronoun}
    return {
        'tokens':
            lit_types.Tokens(parent='text'),
        'coref':
            lit_types.EdgeLabels(align='tokens'),
        'pred_answer':
            lit_types.MulticlassPreds(
                vocab=winogender.ANSWER_VOCAB, parent='answer'),
    }
