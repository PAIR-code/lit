# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from unittest import mock
from absl.testing import absltest
from google.cloud import aiplatform
from vertexai import generative_models
from lit_nlp.examples.gcp import vertexai_models


class ModelsTest(absltest.TestCase):

  @mock.patch(
      "vertexai.generative_models.GenerativeModel.generate_content"
  )
  @mock.patch(
      "vertexai.generative_models.GenerativeModel.__init__",
      return_value=None,
  )
  def test_query_gemini_model(self, mock_init, mock_generate_content):
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

    model = vertexai_models.GeminiFoundationalModel(model_name="gemini-pro")
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

  @mock.patch("google.cloud.aiplatform.models.Endpoint.predict")
  @mock.patch(
      "google.cloud.aiplatform.models.Endpoint.__init__",
      return_value=None,
  )
  def test_query_self_hosted_generative_model(
      self, mock_init, mock_generate_content
  ):
    response1 = aiplatform.models.Prediction(
        predictions=["I say yes you say no"],
        deployed_model_id="",
    )
    response2 = aiplatform.models.Prediction(
        predictions=["I have a dog"],
        deployed_model_id="",
    )
    mock_generate_content.side_effect = [response1, response2]

    model = vertexai_models.SelfHostedGenerativeModel(
        aip_endpoint_name="endpoint_name"
    )
    model._endpoint = mock.MagicMock()
    model._endpoint.predict.side_effect = [response1, response2]

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

    mock_init.assert_called_once_with("endpoint_name")


if __name__ == "__main__":
  absltest.main()
