from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.examples.prompt_debugging import transformers_lms
import numpy as np
from transformers import tokenization_utils

_MAX_LENGTH = 32

_GPT2_PT_PATH = (
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/"
    "gpt2-pt.tar.gz"
)


def _tokenize_text(
    text: str, tokenizer: tokenization_utils.PreTrainedTokenizer
) -> tokenization_utils.BatchEncoding:
  return tokenizer(
      text,
      return_tensors=transformers_lms._HF_PYTORCH,
      add_special_tokens=True,
  )


def _get_text_mean_embeddings(
    text: str, model: transformers_lms.HFBaseModel
) -> np.ndarray:
  tokens = _tokenize_text(text=text, tokenizer=model.tokenizer)
  embeddings = model.embedding_table(tokens["input_ids"]).detach()
  mean_embeddings = np.mean(embeddings.numpy()[0], axis=0)
  return mean_embeddings


class TransformersLMSGeneration(parameterized.TestCase):
  """Test that model classes can predict."""

  def test_gpt2_generation_output(self):
    model = transformers_lms.HFGenerativeModel(
        model_name_or_path=_GPT2_PT_PATH,
        framework=transformers_lms.SUPPORTED_ML_RUNTIMES[0],
        max_length=_MAX_LENGTH,
    )
    model_in = [{"prompt": "Today is"}, {"prompt": "What is the color of"}]
    model_out = list(model.predict(model_in))

    with self.subTest(name="model_input_length_matches_output_length"):
      self.assertLen(model_out, 2)

    with self.subTest(name="model_output_has_expected_spec_keys"):
      expected_output_keys = sorted(model.output_spec().keys())
      for cur_output in model_out:
        self.assertSequenceEqual(
            sorted(cur_output.keys()), expected_output_keys
        )

    with self.subTest(
        name="model_output_prompt_and_response_embeddings_match_those_computed_from_embedding_table"
    ):
      for cur_input, cur_output in zip(model_in, model_out):
        expected_input_embeddings = _get_text_mean_embeddings(
            text=cur_input["prompt"], model=model
        )
        expected_output_embeddings = _get_text_mean_embeddings(
            text=cur_output["response"], model=model
        )
        np.testing.assert_array_almost_equal(
            expected_input_embeddings,
            cur_output["prompt_embeddings"],
        )
        np.testing.assert_array_almost_equal(
            expected_output_embeddings,
            cur_output["response_embeddings"],
        )

  def test_gpt2_batched_generation_has_correct_input_and_output_token_lengths(
      self,
  ):
    model = transformers_lms.HFGenerativeModel(
        model_name_or_path=_GPT2_PT_PATH,
        framework=transformers_lms.SUPPORTED_ML_RUNTIMES[0],
        max_length=_MAX_LENGTH,
    )
    model_in = [{"prompt": "Today is"}, {"prompt": "What is the color of"}]
    batched_outputs = model._get_batched_outputs(model_in)
    tokenized_inputs = [
        _tokenize_text(
            text=input_dict["prompt"],
            tokenizer=model.tokenizer,
        )
        for input_dict in model_in
    ]
    expected_input_token_len = np.array([
        tokenized_input["input_ids"].shape[1]
        for tokenized_input in tokenized_inputs
    ])
    expected_output_token_len = np.full(
        (len(model_in),), _MAX_LENGTH - np.max(expected_input_token_len)
    )
    np.testing.assert_array_equal(
        expected_input_token_len, batched_outputs["ntok_in"]
    )
    np.testing.assert_array_equal(
        expected_output_token_len, batched_outputs["ntok_out"]
    )


if __name__ == "__main__":
  absltest.main()
