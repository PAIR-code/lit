"""🐧 TensorFlow Decision Forests model for the Penguin dataset."""

import copy

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.datasets.penguin_data import VOCABS
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf


class PenguinModel(lit_model.Model):
  """TensorFlow Decision Forest model for penguin classification."""

  def __init__(self, path: str):
    self._model = tf.keras.models.load_model(path)

  def predict_minibatch(self, inputs):

    # Convert LIT input into format model expects.
    def convert_input(inp):
      adjusted_inp = copy.deepcopy(inp)
      island = VOCABS['island'].index(inp['island'])
      sex = VOCABS['sex'].index(inp['sex'])
      species = VOCABS['species'].index(inp['species'])
      adjusted_inp['island'] = island
      adjusted_inp['sex'] = sex
      adjusted_inp['species'] = species
      return adjusted_inp

    adjusted_inputs = [convert_input(inp) for inp in inputs]
    predict_df = pd.DataFrame(adjusted_inputs)
    predict_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        predict_df, label='species')

    model_output = self._model.predict(predict_ds)

    return [{'predicted_species': o} for o in model_output]

  def input_spec(self):
    return {
        'body_mass_g': lit_types.Scalar(),
        'culmen_depth_mm': lit_types.Scalar(),
        'culmen_length_mm': lit_types.Scalar(),
        'flipper_length_mm': lit_types.Scalar(),
        'island': lit_types.CategoryLabel(vocab=VOCABS['island']),
        'sex': lit_types.CategoryLabel(vocab=VOCABS['sex']),
    }

  def output_spec(self):
    return {
        'predicted_species': lit_types.MulticlassPreds(
            parent='species', vocab=VOCABS['species'])
    }
