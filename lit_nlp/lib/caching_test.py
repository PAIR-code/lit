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
from absl.testing import parameterized
from lit_nlp.api import dataset
from lit_nlp.lib import caching
from lit_nlp.lib import testing_utils


class CachingModelWrapperTest(parameterized.TestCase):

  def test_caching_model_wrapper_use_cache(self):
    model = testing_utils.IdentityRegressionModelForTesting()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"val": 1, "_id": "id_to_cache"}]
    results = wrapper.predict(examples)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    results = wrapper.predict(examples)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    self.assertEmpty(wrapper._cache._pred_locks)

  def test_caching_model_wrapper_not_cached(self):
    model = testing_utils.IdentityRegressionModelForTesting()
    wrapper = caching.CachingModelWrapper(model, "test")
    examples = [{"val": 1, "_id": "my_id"}]
    results = wrapper.predict(examples)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    examples = [{"val": 2, "_id": "other_id"}]
    results = wrapper.predict(examples)
    self.assertEqual(2, model.count)
    self.assertEqual({"score": 2}, results[0])

  def test_caching_model_wrapper_uses_cached_subset(self):
    model = testing_utils.IdentityRegressionModelForTesting()
    wrapper = caching.CachingModelWrapper(model, "test")

    examples = [
        {"val": 0, "_id": "zeroth_id"},
        {"val": 1, "_id": "first_id"},
        {"val": 2, "_id": "second_id"},
    ]
    subset = examples[:1]

    # Run the CachingModelWrapper over a subset of examples
    results = wrapper.predict(subset)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 0}, results[0])

    # Now, run the CachingModelWrapper over all of the examples. This should
    # only pass the examples that were not in subset to the wrapped model, and
    # the total number of inputs processed by the wrapped model should be 3
    results = wrapper.predict(examples)
    self.assertEqual(3, model.count)
    self.assertEqual({"score": 0}, results[0])
    self.assertEqual({"score": 1}, results[1])
    self.assertEqual({"score": 2}, results[2])

  @parameterized.named_parameters(
      ("hash_fn=input_hash", caching.input_hash),
      ("hash_fn=custom_fn", lambda x: x["_id"]),
  )
  def test_caching_model_strict_id_validation(
      self, id_hash_fn: dataset.IdFnType
  ):
    model = testing_utils.IdentityRegressionModelForTesting()
    wrapper = caching.CachingModelWrapper(
        model, "test", strict_id_validation=True, id_hash_fn=id_hash_fn
    )
    examples = [{"val": 1, "_id": "b1d0ec818f8aeefdd0551cad96d58e75"}]
    results = wrapper.predict(examples)
    self.assertEqual(1, model.count)
    self.assertEqual({"score": 1}, results[0])
    self.assertEmpty(wrapper._cache._pred_locks)

  def test_caching_model_raises_strict_id_validation_no_id_hash_fn(self):
    model = testing_utils.IdentityRegressionModelForTesting()
    with self.assertRaises(ValueError):
      caching.CachingModelWrapper(model, "test", strict_id_validation=True)

  def test_caching_model_raises_strict_id_validation_differing_ids(self):
    model = testing_utils.IdentityRegressionModelForTesting()
    wrapper = caching.CachingModelWrapper(
        model, "test", strict_id_validation=True, id_hash_fn=caching.input_hash
    )
    examples = [{"val": 1, "_id": "my_id"}]
    with self.assertRaises(ValueError):
      wrapper.predict(examples)


class PredsCacheTest(absltest.TestCase):

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
