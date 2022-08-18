"""Example prompts dataset for use with Dall-E and other text-to-image generative models"""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

class Dalle(lit_dataset.Dataset):
  """Example prompts dataset for use with Dall-E 
  and other text-to-image generative models."""

  def __init__(self):

    prompt = ["A pikachu that looks lika a pug",
              "Trail cam footage of gollum eating watermelon",
              "An astronaut riding a horse in a photorealistic style",
              "Pixar coronavirus movie",
              "Darth Vader in Soviet space propaganda poster",]
    
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