# Copyright 2020 Google LLC
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
# Lint as: python3
"""Tests for lit_nlp.lib.model."""

from absl.testing import absltest

from lit_nlp.lib import caching
from lit_nlp.lib import testing_utils


class CachingTest(absltest.TestCase):

  def test_preds_cache(self):
    """Test with an exact match."""
    cache = caching.PredsCache()
    self.assertEqual("0", cache.info())
    cache.put("test", None)
    self.assertEqual("0", cache.info())
    cache.put("test", ("a", "1"))
    self.assertEqual("1", cache.info())
    self.assertIsNone(None, cache.get(("a", "2")))
    self.assertEqual("test", cache.get(("a", "1")))

  def test_caching_model_wrapper_no_dataset_skip_cache(self):
    model = testing_utils.TestIdentityRegressionModel()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"data": {"val": 1}, "id": "my_id"}]
    results = wrapper.predict_with_metadata(examples)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    results = wrapper.predict_with_metadata(examples)
    self.assertEqual(2, model.count)
    self.assertEqual({"score": 1}, results[0])

  def test_caching_model_wrapper_use_cache(self):
    model = testing_utils.TestIdentityRegressionModel()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"data": {"val": 1}, "id": "id_to_cache"}]
    results = wrapper.predict_with_metadata(examples, "dataset")
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    results = wrapper.predict_with_metadata(examples, "dataset")
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])

  def test_caching_model_wrapper_not_cached(self):
    model = testing_utils.TestIdentityRegressionModel()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"data": {"val": 1}, "id": "my_id"}]
    results = wrapper.predict_with_metadata(examples, "dataset")
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    examples = [{"data": {"val": 2}, "id": "other_id"}]
    results = wrapper.predict_with_metadata(examples)
    self.assertEqual(2, model.count)
    self.assertEqual({"score": 2}, results[0])

  def test_caching_model_wrapper_mixed_list(self):
    model = testing_utils.TestIdentityRegressionModel()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"data": {"val": 1}, "id": "my_id"}]
    results = wrapper.predict_with_metadata(examples, "dataset")
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])

    examples = [
        {
            "data": {
                "val": 0
            },
            "id": "first_id"
        },
        {
            "data": {
                "val": 1
            },
            "id": "my_id"
        },
        {
            "data": {
                "val": 2
            },
            "id": "last_id"
        },
    ]
    results = wrapper.predict_with_metadata(examples, "dataset")
    self.assertEqual(3, model.count)
    self.assertEqual({"score": 0}, results[0])
    self.assertEqual({"score": 1}, results[1])
    self.assertEqual({"score": 2}, results[2])


if __name__ == "__main__":
  absltest.main()
