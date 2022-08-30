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

    Raises:
      AssertionError: if the model's spec does not match that expected for a TyDi
      model.
    """
    # Check inputs
    ispec = model.input_spec()
    self.assertIn("context", ispec)
    self.assertIsInstance(ispec["context"], lit_types.TextSegment)
    if "answers_text" in ispec:
      self.assertIsInstance(ispec["answers_text"], lit_types.MultiSegmentAnnotations)

    # Check outputs
    ospec = model.output_spec()
    self.assertIn("generated_text", ospec)
    self.assertIsInstance(ospec["generated_text"], lit_types.GeneratedText)
    self.assertEqual(ospec["generated_text"].parent, "answers_text")



  def test_TyDimodel(self):
    model = tydi.TyDiModel("TyDiModel", model="dummy", tokenizer="dummy")
    self.validate_TyDiModel(model)  # uses asserts internally



if __name__ == "__main__":
  absltest.main()
