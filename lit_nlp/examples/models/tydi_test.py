"""Tests for lit_nlp.examples.models.tydi."""

from absl.testing import absltest
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import tydi


class TyDiModelTest(absltest.TestCase):
  """Test that model classes conform to the expected spec."""

  def test_model_specs(self):
    model = tydi.TyDiModel("TyDiModel", model="dummy", tokenizer="dummy")
    # Check inputs
    ispec = model.input_spec()
    self.assertIn("context", ispec)
    self.assertIsInstance(ispec["context"], lit_types.TextSegment)
    self.assertIn("answers_text", ispec)
    self.assertIsInstance(
        ispec["answers_text"], lit_types.MultiSegmentAnnotations
    )

    # Check outputs
    ospec = model.output_spec()
    self.assertIn("generated_text", ospec)
    self.assertIsInstance(ospec["generated_text"], lit_types.GeneratedText)
    self.assertEqual(ospec["generated_text"].parent, "answers_text")


if __name__ == "__main__":
  absltest.main()
