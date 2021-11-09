"""Tests for lit_nlp.examples.models.t5."""

from absl.testing import absltest
from lit_nlp.examples.models import t5


class T5ValidationTest(absltest.TestCase):
  """Test that model classes conform to the expected spec."""

  def test_t5hfmodel(self):
    model = t5.T5HFModel("t5-small", model="dummy", tokenizer="dummy")
    model = t5.validate_t5_model(model)  # uses asserts internally

  def test_t5savedmodel(self):
    model = t5.T5SavedModel("/dummy/path", model="dummy")
    model = t5.validate_t5_model(model)  # uses asserts internally


if __name__ == "__main__":
  absltest.main()
