"""Tests for lit_nlp.components.tfx_model."""
import tempfile

from lit_nlp.api import types as lit_types
from lit_nlp.components import tfx_model
import numpy as np
import tensorflow as tf


class TfxModelTest(tf.test.TestCase):

  def setUp(self):
    super(TfxModelTest, self).setUp()
    self._path = tempfile.mkdtemp()
    input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                        name='input_0')
    output_layer = tf.keras.layers.Dense(1, name='output_0')(input_layer)
    model = tf.keras.Model(input_layer, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy)
    model.save(self._path)

  def testTfxModel(self):
    input_spec = {'input_0': lit_types.Scalar()}
    output_spec = {'output_0': lit_types.RegressionScore(parent='input_0')}
    lit_model = tfx_model.TFXModel(self._path,
                                   input_spec=input_spec,
                                   output_spec=output_spec)
    result = list(lit_model.predict([{'input_0': 0.5}]))
    self.assertLen(result, 1)
    result = result[0]
    self.assertListEqual(list(result.keys()), ['output_0'])
    self.assertIsInstance(result['output_0'], np.float32)
    self.assertDictEqual(lit_model.input_spec(), input_spec)
    self.assertDictEqual(lit_model.output_spec(), output_spec)


if __name__ == '__main__':
  tf.test.main()
