"""Tests for serialize."""

import json
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dtypes
from lit_nlp.api import types
from lit_nlp.lib import serialize
import numpy as np


class SerializeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="data_tuple",
          json_dict={
              "__class__": "DataTuple",
              "__name__": "SpanLabel",
              "start": 0,
              "end": 1,
          },
          expected_type=dtypes.SpanLabel,
      ),
      dict(
          testcase_name="empty",
          json_dict={},
          expected_type=dict,
      ),
      dict(
          testcase_name="lit_type",
          json_dict={
              "required": False,
              "annotated": False,
              "default": "",
              "vocab": ["0", "1"],
              "__name__": "CategoryLabel",
          },
          expected_type=types.CategoryLabel,
      ),
      dict(
          testcase_name="nested",
          json_dict={
              "data_tuple": {
                  "__class__": "DataTuple",
                  "__name__": "SpanLabel",
                  "start": 0,
                  "end": 1,
              },
              "lit_type": {
                  "required": False,
                  "annotated": False,
                  "default": "",
                  "vocab": ["0", "1"],
                  "__name__": "CategoryLabel",
              },
              "np_ndarray": {
                  "__class__": "np.ndarray",
                  "__value__": [1, 2, 3],
              },
              "tuple": {
                  "__class__": "tuple",
                  "__value__": [1, 2, 3],
              },
              "vanilla": {
                  "a": 1,
                  "b": "2",
                  "c": [3, 4, 5],
                  "d": True,
              }
          },
          expected_type=dict,
      ),
      dict(
          testcase_name="np_ndarray",
          json_dict={
              "__class__": "np.ndarray",
              "__value__": [1, 2, 3],
          },
          expected_type=np.ndarray,
      ),
      dict(
          testcase_name="tuple",
          json_dict={
              "__class__": "tuple",
              "__value__": [1, 2, 3],
          },
          expected_type=tuple,
      ),
      dict(
          testcase_name="vanilla",
          json_dict={
              "a": 1,
              "b": "2",
              "c": [3, 4, 5],
              "d": True,
          },
          expected_type=dict,
      ),
  )
  def test_from_json(self, json_dict: types.JsonDict, expected_type):
    json_str = json.dumps(json_dict)
    parsed = serialize.from_json(json_str)
    self.assertIsInstance(parsed, expected_type)

  @parameterized.named_parameters(
      ("name_none", {"__name__": None}),
      ("name_number", {"__name__": 3.14159}),
      ("name_invalid", {"__name__": "not_a_lit_type"}),
  )
  def test_from_json_errors(self, json_dict: types.JsonDict):
    with self.assertRaises(serialize.LitJsonParseError):
      json_str = json.dumps(json_dict)
      _ = serialize.from_json(json_str)


if __name__ == "__main__":
  absltest.main()
