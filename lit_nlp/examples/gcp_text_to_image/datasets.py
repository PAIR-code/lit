"""Data loaders for text to image models."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


class TextToImageDataset(lit_dataset.Dataset):
  """TextToImageDataset is a dataset that contains a list of prompts.

  It is used to generate images using the text to image models.
  """

  def __init__(self, prompts: list[str]):
    self._examples = []
    for prompt in prompts:
      self._examples.append({"prompt": prompt})

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {"prompt": lit_types.TextSegment(required=True)}

  def spec(self) -> lit_types.Spec:
    return {"prompt": lit_types.TextSegment()}
