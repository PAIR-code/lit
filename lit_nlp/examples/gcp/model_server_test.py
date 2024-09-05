import os
from unittest import mock

from absl.testing import absltest
from lit_nlp.examples.gcp import model_server
from lit_nlp.examples.prompt_debugging import utils as pd_utils
import webtest


class TestWSGIApp(absltest.TestCase):

  @mock.patch('lit_nlp.examples.prompt_debugging.models.get_models')
  def test_predict_endpoint(self, mock_get_models):
    test_model_name = 'lit_on_gcp_test_model'
    test_model_config = f'{test_model_name}:test_model_path'
    os.environ['MODEL_CONFIG'] = test_model_config

    mock_model = mock.MagicMock()
    mock_model.predict.side_effect = [[{'response': 'test output text'}]]

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

    sal_name, tok_name = pd_utils.generate_model_group_names(test_model_name)

    mock_get_models.return_value = {
        test_model_name: mock_model,
        sal_name: salience_model,
        tok_name: tokenize_model,
    }
    app = webtest.TestApp(model_server.get_wsgi_app())

    response = app.post_json('/predict', {'inputs': 'test_input'})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json, [{'response': 'test output text'}])

    response = app.post_json('/salience', {'inputs': 'test_input'})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(
        response.json,
        [{
            'tokens': ['test', 'output', 'text'],
            'grad_l2': [0.1234, 0.3456, 0.5678],
            'grad_dot_input': [0.1234, -0.3456, 0.5678],
        }],
    )

    response = app.post_json('/tokenize', {'inputs': 'test_input'})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json, [{'tokens': ['test', 'output', 'text']}])


if __name__ == '__main__':
  absltest.main()
