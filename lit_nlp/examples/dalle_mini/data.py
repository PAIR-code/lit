"""Data loaders for dalle-mini model."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


class DallePrompts(lit_dataset.Dataset):

  def __init__(self, prompts: list[str]):
    self.examples = []
    for prompt in prompts:
      self.examples.append({"prompt": prompt})

  def spec(self) -> lit_types.Spec:
    return {"prompt": lit_types.TextSegment()}

  def __iter__(self):
    return iter(self.examples)
