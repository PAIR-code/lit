"""Example prompts to use with Dall-E and other text-to-image GenAI models."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

_CANNED_PROMPTS = (
    "A pikachu that looks lika a pug",
    "Trail cam footage of gollum eating watermelon",
    "An astronaut riding a horse in a photorealistic style",
    "Pixar coronavirus movie",
    "Darth Vader in a Soviet space propaganda poster",
)


class DallePrompts(lit_dataset.Dataset):
  """Example prompts to use with Dall-E and other text-to-image GenAI models."""

  def __init__(self):
    for phrase in _CANNED_PROMPTS:
      self._examples.append({"prompt": phrase})

  def spec(self) -> lit_types.Spec:
    return {"prompt": lit_types.TextSegment()}
