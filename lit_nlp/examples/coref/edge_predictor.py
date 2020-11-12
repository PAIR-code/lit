"""LIT model implementation for single-edge classifier."""
import os
from typing import List, Tuple, Iterable, Optional

from absl import logging
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf

EdgeLabel = lit_dtypes.EdgeLabel
JsonDict = lit_types.JsonDict
Spec = lit_types.Spec


def _extract_span(span: Tuple[int, int], embs: np.ndarray,
                  offsets: np.ndarray) -> np.ndarray:
  start = offsets[span[0]]
  end = offsets[span[1]]
  # <float>[span_length, emb_dim]
  range_embs = embs[start:end]
  # <float>[emb_dim]
  pooled_embs = np.mean(range_embs, axis=0)
  return pooled_embs


def _make_probe_inputs(edges: List[EdgeLabel], embs: np.ndarray,
                       offsets: np.ndarray, src_idx: int):
  for j, edge in enumerate(edges):
    span1_embs = _extract_span(edge.span1, embs, offsets)
    span2_embs = _extract_span(edge.span2, embs, offsets)
    yield {
        'span1_embs': span1_embs,
        'span2_embs': span2_embs,
        'label': edge.label,
        'src_idx': src_idx,
        'edge_idx': j
    }


def _estimate_memory_needs(inputs: List[JsonDict], edge_field: str,
                           output_example: JsonDict):
  """Estimate how much memory is needed to store all activations.

  We store all activations in memory to simplify the implementation, but this
  can grow to be quite large (many GB up to 100s of GB) for large encoders
  and large numbers of targets (such as OntoNotes tasks).

  This will log the estimated memory required, based on the dimension of
  activations and the number of total edges.

  Args:
    inputs: all inputs
    edge_field: name of edges field in input (such as 'coref')
    output_example: a single example from an EdgeFeaturesDataset, used to get
      embedding size
  """
  # Count total edges
  total_edges = sum([len(ex[edge_field]) for ex in inputs])
  # Get embedding size from first output example
  span1_dim = output_example['span1_embs'].size
  span2_dim = output_example['span2_embs'].size
  # pylint: disable=logging-format-interpolation
  logging.warning(
      f'Found {total_edges:d} total edges with embedding dimensions '
      f'{span1_dim:d} + {span2_dim:d} = {(span1_dim+span2_dim):d}.')
  bytes_per_edge = (
      output_example['span1_embs'].nbytes + output_example['span2_embs'].nbytes)
  total_mbytes = total_edges * bytes_per_edge / 1e6
  if total_mbytes < 1e3:
    logging.warning(f'Estimated memory requirement: {total_mbytes:.3G} MB')
  else:
    logging.warning(
        f'Estimated memory requirement: {total_mbytes / 1e3:.3G} GB')
  # pylint: enable=logging-format-interpolation
  # TODO(lit-dev): add factor to account for pointers/overhead and text fields.
  # TODO(lit-dev): check system memory and confirm before proceeding


class EdgeFeaturesDataset(lit_dataset.Dataset):
  """Input examples to a SingleEdgePredictor."""

  def __init__(self, examples):
    self._examples = examples

  @classmethod
  def build(cls,
            inputs: List[JsonDict],
            encoder: lit_model.Model,
            edge_field: str,
            embs_field: str,
            offset_field: str,
            progress=lambda x: x,
            verbose=False):
    """Run encoder and extract span representations for coreference.

    'encoder' should be a model returning one TokenEmbeddings field,
    from which span features will be extracted, as well as a TokenOffsets field
    which maps input tokens to output tokens.

    The returned dataset will contain one example for each label in the inputs'
    EdgeLabels field.

    Args:
      inputs: input Dataset
      encoder: encoder model, compatible with inputs
      edge_field: name of edge field in data
      embs_field: name of embeddings field in model output
      offset_field: name of offset field in model output
      progress: optional pass-through progress indicator
      verbose: if true, print estimated memory usage

    Returns:
      EdgeFeaturesDataset with extracted span representations
    """
    examples = []
    encoder_outputs = progress(encoder.predict(inputs))
    for i, output in enumerate(encoder_outputs):
      exs = _make_probe_inputs(
          inputs[i][edge_field],
          output[embs_field],
          output[offset_field],
          src_idx=i)
      examples.extend(exs)
      if verbose and i == 10:
        _estimate_memory_needs(inputs, edge_field, examples[0])

    return cls(examples)

  def spec(self):
    return {
        'span1_embs': lit_types.Embeddings(),
        'span2_embs': lit_types.Embeddings(),
        'label': lit_types.Scalar(),
        'src_idx': lit_types.Scalar(),
        'edge_idx': lit_types.Scalar(),
    }


class SingleEdgePredictor(lit_model.Model):
  """Coref model for a single edge. Compatible with EdgeFeaturesDataset."""

  def build_model(self, input_dim: int, hidden_dim: int = 256):
    """Construct a Keras model using the Functional API."""
    span1_input = tf.keras.Input(shape=[input_dim])
    span2_input = tf.keras.Input(shape=[input_dim])
    concat_input = tf.keras.layers.Concatenate(axis=-1)(
        [span1_input, span2_input])
    h_repr = tf.keras.layers.Dense(hidden_dim, activation='relu')(concat_input)
    probas = tf.keras.layers.Dense(1, activation='sigmoid')(h_repr)
    return tf.keras.Model(
        inputs={
            'span1_embs': span1_input,
            'span2_embs': span2_input
        },
        outputs=[probas],
        name='coref_model')

  def __init__(self, model_path: Optional[str] = None, **model_kw):
    if model_path:
      # Load from SavedModel
      self.model = tf.keras.models.load_model(model_path)
    else:
      # Construct from params
      self.model = self.build_model(**model_kw)

  def _make_feature_columns(self, inputs: Iterable[JsonDict]):
    """Extract features from input records and return as dict of columns."""
    return {
        'span1_embs':
            tf.constant([ex['span1_embs'] for ex in inputs], dtype=tf.float32),
        'span2_embs':
            tf.constant([ex['span2_embs'] for ex in inputs], dtype=tf.float32),
    }

  def _make_dataset(self, inputs: Iterable[JsonDict]) -> tf.data.Dataset:
    """Make a tf.data.Dataset from inputs in LIT format."""
    # Convert to feature columns for tf.data APIs
    features = self._make_feature_columns(inputs)
    labels = tf.constant([ex['label'] for ex in inputs], dtype=tf.float32)
    return tf.data.Dataset.from_tensor_slices((features, labels))

  def train(self,
            train_inputs: List[JsonDict],
            validation_inputs: List[JsonDict],
            learning_rate=2e-5,
            batch_size=32,
            num_epochs=50,
            keras_callbacks=None):
    """Train an edge classifier."""
    train_dataset = self._make_dataset(train_inputs).shuffle(
        len(train_inputs)).batch(batch_size).repeat(-1)
    # Use larger batch for validation since inference is about 1/2 memory usage
    # of backprop.
    eval_batch_size = 2 * batch_size
    validation_dataset = self._make_dataset(validation_inputs).batch(
        eval_batch_size)

    # Prepare model for training.
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    # TODO(lit-dev): get Keras to train on logits but eval on probas.
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    self.model.compile(optimizer=opt, loss=loss, metrics=[metrics])

    steps_per_epoch = len(train_inputs) // batch_size
    validation_steps = len(validation_inputs) // eval_batch_size
    history = self.model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=keras_callbacks,
        verbose=2)
    return history

  def save(self, path: str):
    """Save model weights.

    To re-load, pass the path to the constructor instead of the name of a
    base model.

    Args:
      path: directory to save to. Will write several files here.
    """
    if not os.path.isdir(path):
      os.mkdir(path)
    self.model.save(path)

  ##
  # LIT API methods
  def max_minibatch_size(self):
    return 128

  def predict_minibatch(self, inputs):
    features = self._make_feature_columns(inputs)
    probas = self.model(features)  # <tf.float32>[batch_size, 1]
    preds = {'proba': tf.squeeze(probas, axis=-1).numpy()}
    return list(utils.unbatch_preds(preds))

  def input_spec(self):
    return {
        'span1_embs': lit_types.Embeddings(),
        'span2_embs': lit_types.Embeddings(),
        'label': lit_types.Scalar(required=False),  # in range [0,1]
    }

  def output_spec(self):
    return {
        'proba':
            lit_types.RegressionScore(parent='label')  # in range [0,1]
    }
