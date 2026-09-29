# Copyright 2026 Google LLC
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
from lit_nlp import app as lit_app
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import testing_utils


class _TestDataset(lit_dataset.Dataset):

  def __init__(self, val: float = 1.0):
    self._examples = [{'val': val}]

  def spec(self) -> lit_types.Spec:
    return {'val': lit_types.Scalar()}

  def init_spec(self) -> lit_types.Spec:
    return {'val': lit_types.Scalar(required=False)}


class AppTest(absltest.TestCase):

  def _make_app(self, demo_mode: bool) -> lit_app.LitApp:
    test_model = testing_utils.IdentityRegressionModelForTesting()
    return lit_app.LitApp(
        models={'test_model': test_model},
        datasets={'test_ds': _TestDataset()},
        generators={},
        interpreters={},
        metrics={},
        client_root=self.create_tempdir().full_path,
        demo_mode=demo_mode,
    )

  def test_create_dataset_blocked_in_demo_mode(self):
    app = self._make_app(demo_mode=True)
    result = app._create_dataset(
        data={'config': {'new_name': 'new_ds', 'val': 2.0}},
        dataset_name='test_ds',
    )
    self.assertIsNone(result)
    self.assertNotIn('new_ds', app._datasets)

  def test_create_dataset_succeeds_when_not_demo_mode(self):
    app = self._make_app(demo_mode=False)
    result = app._create_dataset(
        data={'config': {'new_name': 'new_ds', 'val': 2.0}},
        dataset_name='test_ds',
    )
    self.assertIsNotNone(result)
    _, new_name = result
    self.assertEqual(new_name, 'new_ds')
    self.assertIn('new_ds', app._datasets)

  def test_create_model_blocked_in_demo_mode(self):
    app = self._make_app(demo_mode=True)
    result = app._create_model(
        data={'config': {'new_name': 'new_model'}},
        model_name='test_model',
    )
    self.assertIsNone(result)
    self.assertNotIn('new_model', app._models)

  def test_create_model_succeeds_when_not_demo_mode(self):
    app = self._make_app(demo_mode=False)
    result = app._create_model(
        data={'config': {'new_name': 'new_model'}},
        model_name='test_model',
    )
    self.assertIsNotNone(result)
    _, new_names = result
    self.assertEqual(new_names, ['new_model'])
    self.assertIn('new_model', app._models)


if __name__ == '__main__':
  absltest.main()
