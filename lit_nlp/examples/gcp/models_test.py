"""Tests for lit_nlp.examples.gcp.models."""

from unittest import mock
from absl.testing import absltest
from vertexai import generative_models
from lit_nlp.examples.gcp import models


class ModelsTest(absltest.TestCase):

  @mock.patch(
      "vertexai.generative_models.GenerativeModel.generate_content"
  )
  @mock.patch(
      "vertexai.generative_models.GenerativeModel.__init__",
      return_value=None,
  )
  def test_query_model(self, mock_init, mock_generate_content):
    response1 = generative_models.GenerationResponse.from_dict({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "I say yes you say no"},
                ],
                "role": "model",
            }
        }]
    })
    response2 = generative_models.GenerationResponse.from_dict({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "I have a dog"},
                ],
                "role": "model",
            }
        }]
    })
    mock_generate_content.side_effect = [response1, response2]

    model = models.VertexModelGardenModel(model_name="gemini-pro")
    model._model = mock.MagicMock()
    model._model.generate_content.side_effect = [response1, response2]

    output = model.predict(
        inputs=[{"prompt": "I say yes you say no"}, {"prompt": "I have a dog"}]
    )
    result = list(output)
    self.assertLen(result, 2)
    self.assertEqual(
        result,
        [
            {"response": [("I say yes you say no", None)]},
            {"response": [("I have a dog", None)]},
        ],
    )

    mock_init.assert_called_once_with("gemini-pro")


if __name__ == "__main__":
  absltest.main()
