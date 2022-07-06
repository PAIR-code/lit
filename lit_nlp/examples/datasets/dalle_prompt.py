"""Data loaders for text-to-image model."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import pandas as pd

class Dalle(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self):
    """Dataset constructor, loads the data into memory."""
    data = {'prompt': ["Batman feeding his pet Goldfish"," You're at a museum and there's a painting of a woman wearing a white dress. She has a blue scarf around her neck and she's holding a red rose. You take a picture of the painting, but when you look at the picture, the woman is gone and there's a blue bird in her place.","a kiwi that is both a bird and a fruit"]}
    # into datafrane to decode string
    df = pd.DataFrame(data=data)

    # populate this with data records
    self._examples = [{
      'prompt': row['prompt'],
    } for _, row in df.iterrows()]

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "prompt": lit_types.TextSegment(),
    }