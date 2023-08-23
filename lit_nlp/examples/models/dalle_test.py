"""Tests for lit_nlp.examples.models.dalle."""

from absl.testing import absltest
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import dalle


class DalleModelForTesting(dalle.DalleModel):
  """Dummy model to avoid API keys required to download models during init."""
  def __init__(self, *unused_args, **unused_kw_args):
      pass


class DalleModelTest(absltest.TestCase):
  """Test that model classes conform to the expected spec."""

  def test_model_specs(self):
    model = DalleModelForTesting()
    # Check inputs
    ispec = model.input_spec()
    self.assertIn("prompt", ispec)
    self.assertIsInstance(ispec["prompt"], lit_types.TextSegment)

    # Check outputs
    ospec = model.output_spec()
    self.assertIn("image", ospec)
    self.assertIsInstance(ospec["image"], lit_types.ImageBytesList)
    self.assertIn("clip_score", ospec)
    clip_score_field = ospec["clip_score"]
    self.assertIsInstance(clip_score_field, lit_types.GeneratedTextCandidates)
    self.assertEqual(clip_score_field.parent, "prompt")


if __name__ == "__main__":
  absltest.main()
