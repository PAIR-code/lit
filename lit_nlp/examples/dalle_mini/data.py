"""Data loaders for dalle-mini model."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


class DallePrompts(lit_dataset.Dataset):
  """DallePrompts is a dataset that contains a list of prompts.

  It is used to generate images using the dalle-mini model.
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
