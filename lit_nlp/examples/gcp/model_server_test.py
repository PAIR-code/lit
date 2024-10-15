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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.examples.gcp import constants as lit_gcp_constants
from lit_nlp.examples.gcp import model_server
from lit_nlp.examples.prompt_debugging import utils as pd_utils
import webtest


class TestWSGIApp(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    test_model_name = 'lit_on_gcp_test_model'
    sal_name, tok_name = pd_utils.generate_model_group_names(test_model_name)
    test_model_config = f'{test_model_name}:test_model_path'
    os.environ['MODEL_CONFIG'] = test_model_config

    generation_model = mock.MagicMock()
    generation_model.predict.side_effect = [[{'response': 'test output text'}]]

    salience_model = mock.MagicMock()
    salience_model.predict.side_effect = [[{
        'tokens': ['test', 'output', 'text'],
        'grad_l2': [0.1234, 0.3456, 0.5678],
        'grad_dot_input': [0.1234, -0.3456, 0.5678],
    }]]

    tokenize_model = mock.MagicMock()
    tokenize_model.predict.side_effect = [
        [{'tokens': ['test', 'output', 'text']}]
    ]

    cls.mock_models = {
        test_model_name: generation_model,
        sal_name: salience_model,
        tok_name: tokenize_model,
    }

  @parameterized.named_parameters(
      dict(
          testcase_name=lit_gcp_constants.LlmHTTPEndpoints.GENERATE.value,
          endpoint=f'/{lit_gcp_constants.LlmHTTPEndpoints.GENERATE.value}',
          expected=[{'response': 'test output text'}],
      ),
      dict(
          testcase_name=lit_gcp_constants.LlmHTTPEndpoints.SALIENCE.value,
          endpoint=f'/{lit_gcp_constants.LlmHTTPEndpoints.SALIENCE.value}',
          expected=[{
              'tokens': ['test', 'output', 'text'],
              'grad_l2': [0.1234, 0.3456, 0.5678],
              'grad_dot_input': [0.1234, -0.3456, 0.5678],
          }],
      ),
      dict(
          testcase_name=lit_gcp_constants.LlmHTTPEndpoints.TOKENIZE.value,
          endpoint=f'/{lit_gcp_constants.LlmHTTPEndpoints.TOKENIZE.value}',
          expected=[{'tokens': ['test', 'output', 'text']}],
      ),
  )
  @mock.patch('lit_nlp.examples.prompt_debugging.models.get_models')
  def test_endpoint(self, mock_get_models, endpoint, expected):
    mock_get_models.return_value = self.mock_models
    app = webtest.TestApp(model_server.get_wsgi_app())

    response = app.post_json(endpoint, {'inputs': [{'prompt': 'test input'}]})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json, expected)


if __name__ == '__main__':
  absltest.main()
