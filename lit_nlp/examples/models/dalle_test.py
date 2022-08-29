"""Tests for lit_nlp.examples.models.dalle."""

from absl.testing import absltest
from lit_nlp.examples.models import dalle
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types


class dalle_mini_validation_test(absltest.TestCase):

  """Test that model classes conform to the expected spec."""
  
  def validate_dalle_mini_model(self,model: lit_model.Model):
    """Validate that a given model looks like a dalle mini model.
    Args:
      model: a LIT model
    Raises:
      AssertionError: if the model's spec does not match that expected for a
          dalle mini model.
    """
    # Check inputs
    ispec = model.input_spec()
    assert "prompt" in ispec
    assert isinstance(ispec["prompt"], lit_types.TextSegment)

    # Check outputs
    ospec = model.output_spec()
    assert "image" in ospec
    assert isinstance(
        ospec["image"],
        (lit_types.ImageBytesList))
    assert "clip_score" in ospec
    assert isinstance(
        ospec["clip_score"],
        (lit_types.GeneratedTextCandidates))
    assert ospec["clip_score"].parent == "prompt"



  def test_DalleModel(self):
    model = dalle.DalleModel("dalle-mini/dalle-mini/mini-1:v0", predictions=6)
    self.validate_dalle_mini_model(model)  # uses asserts internally



if __name__ == "__main__":
  absltest.main()