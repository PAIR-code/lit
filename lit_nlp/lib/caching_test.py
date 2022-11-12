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
"""Tests for lit_nlp.lib.model."""

from absl.testing import absltest

from lit_nlp.lib import caching
from lit_nlp.lib import testing_utils


class CachingTest(absltest.TestCase):

  def test_preds_cache(self):
    """Test with an exact match."""
    cache = caching.PredsCache("test")
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
    self.assertEmpty(wrapper._cache._pred_locks)

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

  def test_pred_lock_key(self):
    cache = caching.PredsCache("test")
    cache_key = [("a", "1"), ("a", "2")]

    self.assertIsNone(cache.pred_lock_key(cache_key))

    cache.get_pred_lock(cache_key)
    expected_cache_key = frozenset(cache_key)
    self.assertEqual(expected_cache_key, cache.pred_lock_key(cache_key))

    sub_cache_key = [("a", "1")]
    self.assertEqual(expected_cache_key, cache.pred_lock_key(sub_cache_key))

    mismatch_cache_key = [("b", "1")]
    self.assertIsNone(cache.pred_lock_key(mismatch_cache_key))

  def test_pred_lock_key_no_concurrent_predictions(self):
    cache = caching.PredsCache("test", False)
    cache_key = [("a", "1"), ("a", "2")]

    self.assertIsNone(cache.pred_lock_key(cache_key))

    cache.get_pred_lock(cache_key)
    expected_cache_key = frozenset(
        [caching.PRED_LOCK_KEY_WHEN_NO_CONCURRENT_ACCESS])
    self.assertEqual(expected_cache_key, cache.pred_lock_key(cache_key))

    sub_cache_key = [("a", "1")]
    self.assertEqual(expected_cache_key, cache.pred_lock_key(sub_cache_key))

    mismatch_cache_key = [("b", "1")]
    self.assertEqual(expected_cache_key, cache.pred_lock_key(
        mismatch_cache_key))

  def test_delete_pred_lock(self):
    cache = caching.PredsCache("test")
    cache_key = [("a", "1"), ("a", "2")]

    self.assertIsNone(cache.delete_pred_lock(cache_key))

    lock = cache.get_pred_lock(cache_key)
    self.assertEqual(lock, cache.delete_pred_lock(cache_key))

    self.assertIsNone(cache.delete_pred_lock(cache_key))

  def test_get_pred_lock(self):
    cache = caching.PredsCache("test")
    cache_key = [("a", "1"), ("a", "2")]

    lock = cache.get_pred_lock(cache_key)
    self.assertIsNotNone(lock)

    sub_cache_key = [("a", "2")]
    self.assertEqual(lock, cache.get_pred_lock(sub_cache_key))

    mismatch_cache_key = [("b", "2")]
    self.assertNotEqual(lock, cache.get_pred_lock(mismatch_cache_key))


if __name__ == "__main__":
  absltest.main()
