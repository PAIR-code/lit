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
"""Tests for lit_nlp.lib.utils."""

from absl.testing import absltest

from lit_nlp.api import types
from lit_nlp.lib import utils


class UtilsTest(absltest.TestCase):

  def test_coerce_bool(self):
    self.assertTrue(utils.coerce_bool(True))
    self.assertTrue(utils.coerce_bool(1))
    self.assertTrue(utils.coerce_bool(2.2))
    self.assertTrue(utils.coerce_bool(True))
    self.assertTrue(utils.coerce_bool([0]))
    self.assertTrue(utils.coerce_bool({"a": "hi"}))
    self.assertTrue(utils.coerce_bool("this is true"))

    self.assertFalse(utils.coerce_bool(""))
    self.assertFalse(utils.coerce_bool(0))
    self.assertFalse(utils.coerce_bool("0"))
    self.assertFalse(utils.coerce_bool(False))
    self.assertFalse(utils.coerce_bool("false"))
    self.assertFalse(utils.coerce_bool("False"))
    self.assertFalse(utils.coerce_bool({}))
    self.assertFalse(utils.coerce_bool([]))

  def test_find_keys(self):
    d = {
        "a": True,
        "b": False,
        "c": True
    }
    self.assertEqual(["a", "c"], utils.find_keys(d, lambda a: a))
    self.assertEqual([], utils.find_keys(d, lambda a: a == "nothing"))
    self.assertEqual([], utils.find_keys({}, lambda a: a))

  def test_find_spec_keys(self):
    spec = {
        "score": types.RegressionScore(),
        "scalar_foo": types.Scalar(),
        "text": types.TextSegment(),
        "emb_0": types.Embeddings(),
        "emb_1": types.Embeddings(),
        "tokens": types.Tokens(),
        "generated_text": types.GeneratedText(),
    }
    self.assertEqual(["score"], utils.find_spec_keys(spec,
                                                     types.RegressionScore))
    self.assertEqual(["text", "tokens", "generated_text"],
                     utils.find_spec_keys(spec,
                                          (types.TextSegment, types.Tokens)))
    self.assertEqual(["emb_0", "emb_1"],
                     utils.find_spec_keys(spec, types.Embeddings))
    self.assertEqual([], utils.find_spec_keys(spec, types.AttentionHeads))
    # Check subclasses
    self.assertEqual(
        list(spec.keys()), utils.find_spec_keys(spec, types.LitType))
    self.assertEqual(["text", "generated_text"],
                     utils.find_spec_keys(spec, types.TextSegment))
    self.assertEqual(["score", "scalar_foo"],
                     utils.find_spec_keys(spec, types.Scalar))

  def test_filter_by_keys(self):
    pred = lambda k: k == "a" or k == "b"
    d = {
        "a": True,
        "b": False,
        "c": True
    }
    self.assertDictEqual({"a": True, "b": False}, utils.filter_by_keys(d, pred))

    d2 = {
        "1": True,
        "2": False,
        "3": True
    }
    self.assertDictEqual({}, utils.filter_by_keys(d2, pred))

    self.assertDictEqual({}, utils.filter_by_keys({}, pred))

  def test_copy_and_update(self):
    d = {
        "a": True,
        "b": False,
        "c": True
    }
    update = {
        "a": False,
        "b": True
    }
    expected = {
        "a": False,
        "b": True,
        "c": True
    }
    self.assertDictEqual(expected, utils.copy_and_update(d, update))

    d = {
        "a": True,
        "b": False,
    }
    update = {
        "a": False,
        "c": True
    }
    expected = {
        "a": False,
        "b": False,
        "c": True
    }
    self.assertDictEqual(expected, utils.copy_and_update(d, update))

    d = {
        "a": True,
        "b": False,
    }
    update = {}
    self.assertDictEqual(d, utils.copy_and_update(d, update))

    d = {}
    update = {
        "a": False,
        "c": True
    }
    self.assertDictEqual(update, utils.copy_and_update(d, update))

  def test_remap_dict(self):
    d = {
        "a": True,
        "b": False,
        "c": True
    }
    remap_dict = {
        "a": "a2",
        "b": "b2"
    }
    expected = {
        "a2": True,
        "b2": False,
        "c": True
    }
    self.assertDictEqual(expected, utils.remap_dict(d, remap_dict))

    d = {
        "a": True,
        "b": False,
        "c": True
    }
    remap_dict = {}
    self.assertDictEqual(d, utils.remap_dict(d, remap_dict))

    d = {}
    remap_dict = {
        "a": "a2",
        "b": "b2"
    }
    self.assertDictEqual(d, utils.remap_dict(d, remap_dict))

    d = {
        "a": True,
        "b": False,
        "c": True
    }
    remap_dict = {
        "a": "b",
    }
    expected = {
        "b": False,
        "c": True
    }
    self.assertDictEqual(expected, utils.remap_dict(d, remap_dict))

  def test_find_all_combinations(self):
    l = [1, 2, 3, 4]
    combinations = utils.find_all_combinations(
        l, min_element_count=2, max_element_count=3)
    expected = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 3],
                [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    self.assertListEqual(combinations, expected)

  def test_find_all_combinations_max_is_greater_than_len(self):
    l = [1, 2, 3, 4]
    combinations = utils.find_all_combinations(
        l, min_element_count=2, max_element_count=10)
    expected = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 3],
                [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]]
    self.assertListEqual(combinations, expected)

  def test_find_all_combinations_min_is_greater_than_max(self):
    l = [1, 2, 3, 4]
    combinations = utils.find_all_combinations(
        l, min_element_count=3, max_element_count=2)
    expected = []
    self.assertListEqual(combinations, expected)

  def test_find_all_combinations_min_is_negative(self):
    l = [1, 2, 3, 4]
    combinations = utils.find_all_combinations(
        l, min_element_count=-1, max_element_count=2)
    expected = [[1], [2], [3], [4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4],
                [3, 4]]
    self.assertListEqual(combinations, expected)


if __name__ == "__main__":
  absltest.main()
