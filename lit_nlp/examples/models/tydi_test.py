"""Tests for lit_nlp.examples.models.tydi."""

from absl.testing import absltest
from lit_nlp.examples.models import tydi
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types


class TyDiValidationTest(absltest.TestCase):

  """Test that model classes conform to the expected spec."""
  
  def validate_TyDiModel(self,model: lit_model.Model) -> lit_model.Model:
    """Validate that a given model looks like a TyDi model used by tydi_test.py.
    Args:
      model: a LIT model

    Returns:
      model: the same model

    Raises:
      AssertionError: if the model's spec does not match that expected for a TyDi
      model.
    """
    # Check inputs
    ispec = model.input_spec()
    assert "context" in ispec
    assert isinstance(ispec["context"], lit_types.TextSegment)
    if "answers_text" in ispec:
      assert isinstance(ispec["answers_text"], lit_types.MultiSegmentAnnotations)

    # Check outputs
    ospec = model.output_spec()
    assert "generated_text" in ospec
    assert isinstance(
        ospec["generated_text"],
        (lit_types.GeneratedText))
    assert ospec["generated_text"].parent == "answers_text"

    return model


  def test_TyDimodel(self):
    model = tydi.TyDiModel("TyDiModel", model="dummy", tokenizer="dummy")
    model = self.validate_TyDiModel(model)  # uses asserts internally



if __name__ == "__main__":
  absltest.main()
