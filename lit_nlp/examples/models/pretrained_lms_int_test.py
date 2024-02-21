"""Integration tests for pretrained_lms."""

from absl.testing import absltest
from lit_nlp.examples.models import pretrained_lms


class PretrainedLmsIntTest(absltest.TestCase):
  """Test that model classes can predict."""

  def test_bertmlm(self):
    # Run prediction to ensure no failure.
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/bert-base-uncased.tar.gz"
    model = pretrained_lms.BertMLM(model_path)
    model_in = [{"text": "test text", "tokens": ["test", "[MASK]"]}]
    model_out = list(model.predict(model_in))

    # Sanity-check entries exist in output.
    self.assertLen(model_out, 1)
    self.assertIn("pred_tokens", model_out[0])
    self.assertIn("cls_emb", model_out[0])

  def test_gpt2(self):
    # Run prediction to ensure no failure.
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz"
    model = pretrained_lms.GPT2LanguageModel(model_path)
    model_in = [{"text": "test text"}, {"text": "longer test text"}]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 2)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())

  def test_gpt2_generation(self):
    # Run prediction to ensure no failure.
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz"
    model = pretrained_lms.GPT2GenerativeModel(model_name_or_path=model_path)
    model_in = [{"prompt": "Today is"}, {"prompt": "What is the color of"}]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 2)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())

    # Check that the embedding dimension is the same for prompt and response.
    self.assertEqual(
        model_out[0]["prompt_embeddings"].shape,
        model_out[0]["response_embeddings"].shape,
    )


if __name__ == "__main__":
  absltest.main()
