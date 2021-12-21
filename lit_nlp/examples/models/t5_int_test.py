"""Integration tests for lit_nlp.examples.models.t5."""

from absl.testing import absltest
from lit_nlp.examples.models import t5


class T5IntTest(absltest.TestCase):
  """Test that model can predict."""

  def test_t5_predict(self):
    # Run prediction to ensure no failure.
    model_path = "t5-small"
    model = t5.T5HFModel(model_path, num_to_generate=1, token_top_k=1,
                         output_attention=False)
    model_in = [{"input_text": "test text"}]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 1)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())

if __name__ == "__main__":
  absltest.main()
