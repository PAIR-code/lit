"""Wrapper for using TFX-generated models within LIT."""
from typing import Iterator, List, Text

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
import tensorflow as tf
import tensorflow_text as tf_text  # pylint: disable=unused-import

_SERVING_DEFAULT_SIGNATURE = 'serving_default'


@attr.s(auto_attribs=True)
class TFXModelConfig(object):
  """Configuration object for TFX Models."""
  path: Text
  input_spec: lit_types.Spec
  output_spec: lit_types.Spec
  signature: Text = _SERVING_DEFAULT_SIGNATURE


# TODO(b/188036366): Revisit the assumed mapping between input values and
# TF.Examples.
def _inputs_to_serialized_example(input_dict: lit_types.JsonDict):
  """Converts the input dictionary to a serialized tf example."""
  feature_dict = {}
  for k, v in input_dict.items():
    if not isinstance(v, list):
      v = [v]
    if isinstance(v[0], int):
      feature_dict[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      feature_dict[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    else:
      feature_dict[k] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[bytes(i, 'utf-8') for i in v]))
  result = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return result.SerializeToString()


class TFXModel(lit_model.Model):
  """Wrapper for querying a TFX-generated SavedModel."""

  def __init__(self, config: TFXModelConfig):
    self._model = tf.saved_model.load(config.path)
    self._signature = config.signature
    self._input_spec = config.input_spec
    self._output_spec = config.output_spec

  def predict_minibatch(
      self, inputs: List[lit_types.JsonDict]) -> Iterator[lit_types.JsonDict]:
    for i in inputs:
      filtered_inputs = {k: v for k, v in i.items() if k in self._input_spec}
      result = self._model.signatures[self._signature](
          tf.constant([_inputs_to_serialized_example(filtered_inputs)]))
      result = {
          k: tf.squeeze(v).numpy().tolist()
          for k, v in result.items()
          if k in self._output_spec
      }
      for k, v in result.items():
        # If doing Multiclass Prediction for a Binary Classifier.
        if (isinstance(self._output_spec[k], lit_types.MulticlassPreds) and
            not isinstance(v, list)):
          result[k] = [1 - v, v]
      yield result

  def input_spec(self) -> lit_types.Spec:
    return self._input_spec

  def output_spec(self) -> lit_types.Spec:
    return self._output_spec
