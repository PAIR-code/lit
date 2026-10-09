"""🐧 Scikit-learn model for the Penguin dataset."""

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.penguin import data as penguin_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_VOCABS = penguin_data.VOCABS


class PenguinModel(lit_model.BatchedModel):
  """Scikit-learn model for penguin classification.

  The model is trained on the penguin dataset at construction time; there
  is no external model file to download.
  """

  def __init__(self):
    examples = penguin_data.PenguinDataset().examples
    features = np.array([self._convert_input(ex) for ex in examples])
    labels = [ex['species'] for ex in examples]
    self.model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    self.model.fit(features, labels)

  def _convert_input(self, inp):
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
    return ex

  def max_minibatch_size(self) -> int:
    return 32

  def predict_minibatch(self, inputs):
    adjusted_inputs = np.array([self._convert_input(inp) for inp in inputs])
    model_output = self.model.predict_proba(adjusted_inputs)

    # sklearn sorts classes alphabetically, which matches VOCABS['species'].
    assert list(self.model.classes_) == _VOCABS['species']
    ret = [{'predicted_species': out} for out in model_output]
    return ret

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {}

  def input_spec(self) -> lit_types.Spec:
    return penguin_data.INPUT_SPEC

  def output_spec(self) -> lit_types.Spec:
    return {
        'predicted_species': lit_types.MulticlassPreds(
            parent='species', vocab=_VOCABS['species']
        )
    }
