"""Integration tests for pretrained_lms."""

from absl.testing import absltest
from lit_nlp.examples.models import pretrained_lms


class PretrainedLmsIntTest(absltest.TestCase):
  """Test that model classes can predict."""

  def test_bertmlm(self):
    # Run prediction to ensure no failure.
    model_path = "bert-base-uncased"
    model = pretrained_lms.BertMLM(model_path)
    model_in = [{"text": "test text", "tokens": ["test", "[MASK]"]}]
    model_out = list(model.predict(model_in))

    # Sanity-check entries exist in output.
    self.assertLen(model_out, 1)
    self.assertIn("pred_tokens", model_out[0])
    self.assertIn("cls_emb", model_out[0])

  def test_gpt2(self):
    # Run prediction to ensure no failure.
    model_path = "gpt2"
    model = pretrained_lms.GPT2LanguageModel(model_path)
    model_in = [{"text": "test text"}]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 1)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())

if __name__ == "__main__":
  absltest.main()
