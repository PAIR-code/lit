"""Tests for lit_nlp.examples.models.tydi."""

from absl.testing import absltest
from lit_nlp.examples.models import tydi


class TyDiValidationTest(absltest.TestCase):
  """Test that model classes conform to the expected spec."""

  def test_TyDimodel(self):
    model = tydi.TyDiModel("TyDiModel", model="dummy", tokenizer="dummy")
    model = tydi.validate_TyDiModel(model)  # uses asserts internally



if __name__ == "__main__":
  absltest.main()
