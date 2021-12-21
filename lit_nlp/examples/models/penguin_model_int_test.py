"""Integration tests for penguin_model."""

from absl.testing import absltest
from lit_nlp.examples.models import penguin_model
import transformers


class PenguinModelIntTest(absltest.TestCase):
  """Test that model class can predict."""

  def test_model(self):
    # Create model.
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/penguin.h5"  # pylint: disable=line-too-long
    if model_path.endswith(".h5"):
      model_path = transformers.file_utils.cached_path(model_path)
    model = penguin_model.PenguinModel(model_path)

    # Run prediction to ensure no failure.
    model_in = [{
        "body_mass_g": 4000,
        "culmen_depth_mm": 15,
        "culmen_length_mm": 50,
        "flipper_length_mm": 200,
        "island": "Biscoe",
        "sex": "Male",
    }]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 1)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())


if __name__ == "__main__":
  absltest.main()
