"""Example prompts dataset for use with Dall-E and other text-to-image generative models"""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

class Dalle(lit_dataset.Dataset):
  """Example prompts dataset for use with Dall-E 
  and other text-to-image generative models."""

  def __init__(self):

    prompt = ["starbucks in the hobbit shire",
              "trail cam footage of gollum eating watermelon",
              "two cats in hazmat suit"]
    
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