"""Example prompts dataset for use with Dall-E and other text-to-image generative models"""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

class Dalle(lit_dataset.Dataset):
  """Example prompts dataset for use with Dall-E 
  and other text-to-image generative models."""

  def __init__(self):

    # just keeping one prompt for testing for time being
    prompt = ["Airbus beluga with beluga whale",
              "two cats in hazmat suit cooking"]
    
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