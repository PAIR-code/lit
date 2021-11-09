"""Tests for lit_nlp.components.tfx_model."""
import tempfile

from lit_nlp.api import types as lit_types
from lit_nlp.components import tfx_model
import tensorflow as tf


class TfxModelTest(tf.test.TestCase):

  def setUp(self):
    super(TfxModelTest, self).setUp()
    self._path = tempfile.mkdtemp()
    input_layer = tf.keras.layers.Input(
        shape=(1), dtype=tf.string, name='example')
    parsed_input = tf.io.parse_example(
        tf.reshape(input_layer, [-1]),
        {'input_0': tf.io.FixedLenFeature([1], dtype=tf.float32)})
    output_layer = tf.keras.layers.Dense(
        1, name='output_0')(
            parsed_input['input_0'])
    model = tf.keras.Model(input_layer, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy)
    model.save(self._path)

  def testTfxModel(self):
    input_spec = {'input_0': lit_types.Scalar()}
    output_spec = {
        'output_0':
            lit_types.MulticlassPreds(vocab=['0', '1'], parent='input_0')
    }
    config = tfx_model.TFXModelConfig(self._path, input_spec, output_spec)
    lit_model = tfx_model.TFXModel(config)
    result = list(lit_model.predict([{'input_0': 0.5}]))
    self.assertLen(result, 1)
    result = result[0]
    self.assertListEqual(list(result.keys()), ['output_0'])
    self.assertLen(result['output_0'], 2)
    self.assertIsInstance(result['output_0'][0], float)
    self.assertIsInstance(result['output_0'][1], float)
    self.assertDictEqual(lit_model.input_spec(), input_spec)
    self.assertDictEqual(lit_model.output_spec(), output_spec)


if __name__ == '__main__':
  tf.test.main()
