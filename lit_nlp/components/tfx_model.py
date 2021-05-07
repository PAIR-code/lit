"""Wrapper for using TFX-generated models within LIT."""
from typing import List, Iterator

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

import tensorflow as tf

_SERVING_DEFAULT_SIGNATURE = 'serving_default'


class TFXModel(lit_model.Model):
  """Wrapper for querying a TFX-generated SavedModel."""

  def __init__(self, path: str, input_spec: lit_types.Spec,
               output_spec: lit_types.Spec,
               signature: str = _SERVING_DEFAULT_SIGNATURE):
    self._model = tf.saved_model.load(path)
    self._signature = signature
    self._input_spec = input_spec
    self._output_spec = output_spec

  def predict_minibatch(self, inputs: List[lit_types.JsonDict]) -> Iterator[
      lit_types.JsonDict]:
    for i in inputs:
      result = self._model.signatures[self._signature](
          **{k: tf.reshape(tf.constant(v), [1, -1]) for k, v in i.items()})
      result = {k: tf.squeeze(v).numpy() for k, v in result.items()}
      yield result

  def input_spec(self) -> lit_types.Spec:
    return self._input_spec

  def output_spec(self) -> lit_types.Spec:
    return self._output_spec
