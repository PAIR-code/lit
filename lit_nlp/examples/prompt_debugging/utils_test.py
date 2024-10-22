from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.examples.prompt_debugging import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty_name",
          name="",
          expected_salience_name="__salience",
          expected_tokenizer_name="__tokenizer",
      ),
      dict(
          testcase_name="known_name",
          name="gemma",
          expected_salience_name="_gemma_salience",
          expected_tokenizer_name="_gemma_tokenizer",
      ),
      dict(
          testcase_name="custom_name_with_spaces",
          name="my model",
          expected_salience_name="_my model_salience",
          expected_tokenizer_name="_my model_tokenizer",
      ),
  )
  def test_generate_model_group_names(
      self, name, expected_salience_name, expected_tokenizer_name
  ):
    salience_name, tokenizer_name = utils.generate_model_group_names(name)
    self.assertEqual(salience_name, expected_salience_name)
    self.assertEqual(tokenizer_name, expected_tokenizer_name)


if __name__ == "__main__":
  absltest.main()
