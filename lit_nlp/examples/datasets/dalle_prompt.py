"""Data loaders for summarization datasets."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds
import pandas as pd

class Dalle(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self):
    """Dataset constructor, loads the data into memory."""
    # data = {'prompt': ["sunset over a lake in the mountains","the Eiffel tower landing on the moon","Crypto Bro working at McDonalds"]}
    data = {'prompt': ["the Eiffel tower landing on the moon"]}
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