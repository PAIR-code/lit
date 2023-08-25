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
"""Tests for lit_nlp.lib.utils."""

from collections.abc import Callable, Sequence
import copy
from typing import Any, Optional, TypeVar, Union

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np

_BATCHING_RECORDS: Sequence[types.JsonDict] = [
    {"foo": 1, "bar": "one"},
    {"foo": 2, "bar": "two"},
    {"foo": 3, "bar": "three"},
]

_VALIDATION_SPEC: types.Spec = {
    "required_scalar": types.Scalar(),
    "required_text_segment": types.String(),
    "optional_boolean": types.Boolean(required=False),
}

T = TypeVar("T")


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bool", True),
      ("bool_as_javascript_str", "true"),
      ("bool_as_str", "True"),
      ("non_zero_int", 1),
      ("non_zero_float", -2.2),
      ("non_empty_dict", {"a": "hi"}),
      ("non_empty_list", [0]),
      ("non_empty_string", "this is true"),
  )
  def test_coerce_bool_true(self, value: Any):
    self.assertTrue(utils.coerce_bool(value))

  @parameterized.named_parameters(
      ("bool", False),
      ("bool_as_javascript_str", "false"),
      ("bool_as_str", "False"),
      ("zero", 0),
      ("zero_as_str", "0"),
      ("empty_dict", {}),
      ("empty_list", []),
      ("empty_str", ""),
  )
  def test_coerce_bool_false(self, value: Any):
    self.assertFalse(utils.coerce_bool(value))

  @parameterized.named_parameters(
      dict(
          testcase_name="with_truthy_values",
          d={"a": True, "b": False, "c": True},
          predicate=lambda a: a,
          expected=["a", "c"],
      ),
      dict(
          testcase_name="with_specific_value_missing",
          d={"a": True, "b": False, "c": True},
          predicate=lambda a: a == "nothing",
          expected=[],
      ),
      dict(
          testcase_name="where_d_is_empty",
          d={},
          predicate=lambda a: a,
          expected=[],
      ),
  )
  def test_find_keys(
      self,
      d: dict[str, bool],
      predicate: Callable[[utils.V], bool],
      expected: Sequence[str],
  ):
    found = utils.find_keys(d, predicate)
    self.assertEqual(expected, found)

  @parameterized.named_parameters(
      dict(
          testcase_name="all",
          types_to_find=types.LitType,
          expected=[
              "score",
              "scalar_foo",
              "text",
              "emb_0",
              "emb_1",
              "tokens",
              "generated_text",
          ],
      ),
      dict(
          testcase_name="attention_heads",
          types_to_find=types.AttentionHeads,
          expected=[],
      ),
      dict(
          testcase_name="embeddings",
          types_to_find=types.Embeddings,
          expected=["emb_0", "emb_1"],
      ),
      dict(
          testcase_name="regression_only",
          types_to_find=types.RegressionScore,
          expected=["score"],
      ),
      dict(
          testcase_name="scalars",  # includes RegressionScore subclass
          types_to_find=types.Scalar,
          expected=["score", "scalar_foo"],
      ),
      dict(
          testcase_name="text_segement",  # Includes GeneratedText subclass
          types_to_find=types.TextSegment,
          expected=["text", "generated_text"],
      ),
      dict(
          testcase_name="text_like_tuple",
          types_to_find=(types.TextSegment, types.Tokens),
          expected=["text", "tokens", "generated_text"],
      ),
  )
  def test_find_spec_keys(
      self,
      types_to_find: Union[types.LitType, Sequence[types.LitType]],
      expected: Sequence[str],
  ):
    spec = {
        "score": types.RegressionScore(),
        "scalar_foo": types.Scalar(),
        "text": types.TextSegment(),
        "emb_0": types.Embeddings(),
        "emb_1": types.Embeddings(),
        "tokens": types.Tokens(),
        "generated_text": types.GeneratedText(),
    }
    found = utils.find_spec_keys(spec, types_to_find)
    self.assertEqual(expected, found)

  @parameterized.named_parameters(
      dict(
          testcase_name="d_contains_predicate_keys",
          d={"a": True, "b": False, "c": True},
          expected={"a": True, "b": False},
      ),
      dict(
          testcase_name="d_is_empty",
          d={},
          expected={},
      ),
      dict(
          testcase_name="d_missing_predicate_keys",
          d={"1": True, "2": False, "3": True},
          expected={},
      ),
  )
  def test_filter_by_keys(
      self, d: dict[utils.K, utils.V], expected: dict[utils.K, utils.V]
  ):
    predicate = lambda k: k in ("a", "b")
    filtered = utils.filter_by_keys(d, predicate)
    self.assertEqual(expected, filtered)

  @parameterized.named_parameters(
      dict(
          testcase_name="keys_is_empty",
          keys=[],
          expected={},
      ),
      dict(
          testcase_name="keys_is_None",
          keys=None,
          expected={"foo": [1, 2, 3], "bar": ["one", "two", "three"]},
      ),
      dict(
          testcase_name="keys_are_a_subset",
          keys=["bar"],
          expected={"bar": ["one", "two", "three"]},
      ),
      dict(
          testcase_name="keys_are_the_totality",
          keys=["foo", "bar"],
          expected={"foo": [1, 2, 3], "bar": ["one", "two", "three"]},
      ),
  )
  def test_batch_inputs(
      self, keys: Optional[list[str]], expected: dict[str, list[Any]]
  ):
    batched = utils.batch_inputs(_BATCHING_RECORDS, keys=keys)
    self.assertEqual(expected, batched)

  @parameterized.named_parameters(
      ("AssertionError_for_empty_inputs", [], None, AssertionError),
      ("KeyError_for_disjoint_keys", _BATCHING_RECORDS, ["baz"], KeyError),
  )
  def test_batch_inputs_raises(
      self,
      inputs: Sequence[types.JsonDict],
      keys: Optional[list[str]],
      expected: Exception,
  ):
    with self.assertRaises(expected):
      utils.batch_inputs(inputs, keys=keys)

  @parameterized.named_parameters(
      dict(
          testcase_name="pad_to_end",
          inputs=[1, 2, 3],
          min_len=5,
          pad_val=0,
          expected=[1, 2, 3, 0, 0],
      ),
      dict(
          testcase_name="pad_length_exact",
          inputs=[1, 2, 3],
          min_len=3,
          pad_val=0,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name="pad_too_long",
          inputs=[1, 2, 3, 4, 5],
          min_len=3,
          pad_val=0,
          expected=[1, 2, 3, 4, 5],
      ),
      dict(
          testcase_name="pad_with_strings",
          inputs=["one", "two", "three"],
          min_len=5,
          pad_val="",
          expected=["one", "two", "three", "", ""],
      ),
  )
  def test_pad1d(
      self, inputs: list[T], min_len: T, pad_val: T, expected: list[T]
  ):
    self.assertEqual(utils.pad1d(inputs, min_len, pad_val), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="remap_to_new_names",
          d={"a": True, "b": False, "c": True},
          keymap={"a": "a2", "b": "b2"},
          expected={"a2": True, "b2": False, "c": True},
      ),
      dict(
          testcase_name="remap_to_existing_name",
          d={"a": True, "b": False, "c": True},
          keymap={"a": "b"},
          expected={"b": False, "c": True},
      ),
      dict(
          testcase_name="keymap_is_empty",
          d={"a": True, "b": False, "c": True},
          keymap={},
          expected={"a": True, "b": False, "c": True},
      ),
      dict(
          testcase_name="dict_is_empty",
          d={},
          keymap={"a": "a2", "b": "b2"},
          expected={},
      ),
  )
  def test_remap_dict(
      self,
      d: dict[utils.K, utils.V],
      keymap: dict[utils.K, utils.K],
      expected: dict[utils.K, utils.V],
  ):
    remapped = utils.remap_dict(d, keymap)
    self.assertEqual(expected, remapped)

  @parameterized.named_parameters(
      dict(
          testcase_name="min_and_max_within_bounds",
          min_element_count=2,
          max_element_count=3,
          expected=[
              [1, 2],
              [1, 3],
              [1, 4],
              [2, 3],
              [2, 4],
              [3, 4],
              [1, 2, 3],
              [1, 2, 4],
              [1, 3, 4],
              [2, 3, 4],
          ],
      ),
      dict(
          testcase_name="max_is_greater_than_len",
          min_element_count=2,
          max_element_count=10,
          expected=[
              [1, 2],
              [1, 3],
              [1, 4],
              [2, 3],
              [2, 4],
              [3, 4],
              [1, 2, 3],
              [1, 2, 4],
              [1, 3, 4],
              [2, 3, 4],
              [1, 2, 3, 4],
          ],
      ),
      dict(
          testcase_name="min_is_greater_than_max",
          min_element_count=3,
          max_element_count=2,
          expected=[],
      ),
      dict(
          testcase_name="min_is_negative",
          min_element_count=-1,
          max_element_count=2,
          expected=[
              [1],
              [2],
              [3],
              [4],
              [1, 2],
              [1, 3],
              [1, 4],
              [2, 3],
              [2, 4],
              [3, 4],
          ],
      ),
  )
  def test_find_all_combinations(
      self,
      min_element_count: int,
      max_element_count: int,
      expected: list[list[int]],
  ):
    combinations = utils.find_all_combinations(
        [1, 2, 3, 4], min_element_count, max_element_count
    )
    self.assertEqual(combinations, expected)

  def test_get_real(self):
    l = np.array([1, 2, 3, 4])
    self.assertListEqual(utils.coerce_real(l).tolist(), l.tolist())

    l = np.array([1, 2 + 0.5j, 3, 4])
    self.assertListEqual(utils.coerce_real(l, 0.51).tolist(), [1, 2, 3, 4])

    with self.assertRaises(AssertionError):
      utils.coerce_real(l, 0.4)

  @parameterized.named_parameters(
      dict(
          testcase_name="with_all_params",
          config={
              "required_scalar": 0,
              "required_text_segment": "test",
              "optional_boolean": True,
          },
      ),
      dict(
          testcase_name="with_extra_params",
          config={
              "required_scalar": 0,
              "required_text_segment": "test",
              "optional_boolean": True,
              "param_not_in_spec": True,
          },
      ),
      dict(
          testcase_name="with_only_required_params",
          config={
              "required_scalar": 0,
              "required_text_segment": "test",
          },
      ),
  )
  def test_validate_config_against_spec(self, config: types.JsonDict):
    validated = utils.validate_config_against_spec(
        config, _VALIDATION_SPEC, "unittest"
    )
    self.assertIs(config, validated)

  @parameterized.named_parameters(
      dict(
          testcase_name="for_missing_params",
          config={
              "required_scalar": 0,
              "optional_boolean": True,
          },
      ),
      dict(
          testcase_name="for_unsupported_params",
          config={
              "required_scalar": 0,
              "required_text_segment": "test",
              "param_not_in_spec": True,
          },
      ),
  )
  def test_validate_config_against_spec_raises(self, config: types.JsonDict):
    with self.assertRaises(KeyError):
      utils.validate_config_against_spec(
          config, _VALIDATION_SPEC, "unittest", raise_for_unsupported=True
      )

  def test_combine_specs_not_overlapping(self):
    spec1 = {"string": types.String()}
    spec2 = {"int": types.Integer()}
    spec = utils.combine_specs(spec1, spec2)
    self.assertEqual(spec, {"string": types.String(), "int": types.Integer()})

  def test_combine_specs_overlapping_and_compatible(self):
    spec1 = {"string": types.String()}
    spec2 = {"int": types.Integer(), "string": types.String()}
    spec = utils.combine_specs(spec1, spec2)
    self.assertEqual(spec, {"string": types.String(), "int": types.Integer()})

  def test_combine_specs_conflicting(self):
    spec1 = {"string": types.String()}
    spec2 = {"string": types.TextSegment()}
    with self.assertRaises(ValueError):
      utils.combine_specs(spec1, spec2)

  def test_make_modified_input_one_field(self):
    ex = {
        "foo": 123,
        "bar": 234,
        "_id": "a1b2c3",
        "_meta": {"parentId": "000000"},
    }
    copy_of_original = copy.deepcopy(ex)
    new_ex = utils.make_modified_input(ex, {"bar": 345}, "testFn")
    expected = {
        "foo": 123,
        "bar": 345,
        "_id": "",
        "_meta": {"parentId": "a1b2c3", "added": True, "source": "testFn"},
    }
    self.assertEqual(new_ex, expected)
    # Check that original is unchanged
    self.assertEqual(ex, copy_of_original)

  def test_make_modified_input_two_fields(self):
    ex = {
        "foo": 123,
        "bar": 234,
        "_id": "a1b2c3",
        "_meta": {"parentId": "000000"},
    }
    new_ex = utils.make_modified_input(ex, {"foo": 234, "bar": 345}, "testFn")
    expected = {
        "foo": 234,
        "bar": 345,
        "_id": "",
        "_meta": {"parentId": "a1b2c3", "added": True, "source": "testFn"},
    }
    self.assertEqual(new_ex, expected)

  def test_make_modified_input_new_field(self):
    ex = {
        "foo": 123,
        "bar": 234,
        "_id": "a1b2c3",
        "_meta": {"parentId": "000000"},
    }
    new_ex = utils.make_modified_input(ex, {"baz": "spam and eggs"}, "testFn")
    expected = {
        "foo": 123,
        "bar": 234,
        "baz": "spam and eggs",
        "_id": "",
        "_meta": {"parentId": "a1b2c3", "added": True, "source": "testFn"},
    }
    self.assertEqual(new_ex, expected)

  def test_make_modified_input_unmodified(self):
    ex = {
        "foo": 123,
        "bar": 234,
        "_id": "a1b2c3",
        "_meta": {"parentId": "000000"},
    }
    copy_of_original = copy.deepcopy(ex)
    new_ex = utils.make_modified_input(ex, {"foo": 123, "bar": 234}, "testFn")
    self.assertEqual(new_ex, copy_of_original)
    self.assertIs(new_ex, ex)  # same object back

  def test_make_modified_input_empty_overrides(self):
    ex = {
        "foo": 123,
        "bar": 234,
        "_id": "a1b2c3",
        "_meta": {"parentId": "000000"},
    }
    copy_of_original = copy.deepcopy(ex)
    new_ex = utils.make_modified_input(ex, {}, "testFn")
    self.assertEqual(new_ex, copy_of_original)
    self.assertIs(new_ex, ex)  # same object back

  def test_make_modified_input_not_indexed(self):
    ex = {
        "foo": 123,
        "bar": 234,
    }
    copy_of_original = copy.deepcopy(ex)
    new_ex = utils.make_modified_input(ex, {"bar": 345}, "testFn")
    expected = {
        "foo": 123,
        "bar": 345,
    }
    self.assertEqual(new_ex, expected)
    # Check that original is unchanged
    self.assertEqual(ex, copy_of_original)


if __name__ == "__main__":
  absltest.main()
