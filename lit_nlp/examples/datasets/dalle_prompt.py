"""Data loaders for text-to-image model."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd

class Dalle(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self):

    prompt = ["A still of Homer Simpson in The Blair Witch Project",
              "A still of Homer Simpson in Jaws",
              "Mario playing himself at an arcade"]

    # populate this with data records
    for phrase in prompt:
      self._examples.append({
          "prompt": phrase
      })

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "prompt": lit_types.TextSegment(),
    }