from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.examples.prompt_debugging import transformers_lms


class TransformersLMSGeneration(parameterized.TestCase):
  """Test that model classes can predict."""

  @parameterized.named_parameters(
      dict(
          testcase_name="tensorflow",
          framework=transformers_lms.MLFramework.TF.value,
          model_path="https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2.tar.gz",
      ),
      dict(
          testcase_name="pytorch",
          framework=transformers_lms.MLFramework.PT.value,
          model_path="https://storage.googleapis.com/what-if-tool-resources/lit-models/gpt2-pt.tar.gz",
      ),
  )
  def test_gpt2_generation(self, framework, model_path):
    model = transformers_lms.HFGenerativeModel(
        model_name_or_path=model_path, framework=framework
    )
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
