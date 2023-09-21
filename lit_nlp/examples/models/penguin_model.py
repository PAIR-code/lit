"""🐧 TensorFlow Keras model for the Penguin dataset."""

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.datasets import penguin_data
from lit_nlp.lib import file_cache
import numpy as np
import tensorflow as tf

_VOCABS = penguin_data.VOCABS


class PenguinModel(lit_model.BatchedModel):
  """TensorFlow Keras model for penguin classification."""

  def __init__(self, path: str):
    path = file_cache.cached_path(path)
    self.model = tf.keras.models.load_model(path)
    # Feature column means and variance to normalize values before
    # prediction.
    self.means = np.array([
        4.23795547e+03, 1.71222672e+01, 4.40004048e+01, 2.01587045e+02,
        5.14170040e-01, 3.40080972e-01, 1.45748988e-01, 4.89878543e-01,
        5.10121457e-01
    ])
    self.vars = np.array([
        6.29523451e+05, 4.17662973e+00, 2.76167220e+01, 1.97108820e+02,
        2.49799210e-01, 2.24425904e-01, 1.24506220e-01, 2.49897556e-01,
        2.49897556e-01
    ])

  def max_minibatch_size(self) -> int:
    return 32

  def predict_minibatch(self, inputs):

    def convert_input(inp):
      ex = np.array([
          inp['body_mass_g'], inp['culmen_depth_mm'], inp['culmen_length_mm'],
          inp['flipper_length_mm'], 0, 0, 0, 0, 0
      ])
      # Set one-hot encodings of categorical features.
      island_index = _VOCABS['island'].index(inp['island'])
      sex_index = _VOCABS['sex'].index(inp['sex'])
      # Island one-hot encodings start at input index 4.
      ex[island_index + 4] = 1
      # Sex one-hot encodings start at input index 7.
      ex[sex_index + 7] = 1

      # Normalize feature values.
      ex = (ex - self.means) / (self.vars**0.5)
      return ex.tolist()

    adjusted_inputs = [convert_input(inp) for inp in inputs]
    model_output = self.model.predict(adjusted_inputs)

    ret = [{'predicted_species': out} for out in model_output]
    return ret

  @classmethod
  def init_spec(cls) -> lit_types.Spec:

    default_model_path = ('https://storage.googleapis.com/'
                          'what-if-tool-resources/lit-models/penguin.h5')
    return {'path': lit_types.String(default=default_model_path)}

  def input_spec(self) -> lit_types.Spec:
    return penguin_data.INPUT_SPEC

  def output_spec(self) -> lit_types.Spec:
    return {
        'predicted_species': lit_types.MulticlassPreds(
            parent='species', vocab=_VOCABS['species']
        )
    }
