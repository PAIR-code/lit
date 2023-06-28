"""Tests for validation."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset
from lit_nlp.api import types
from lit_nlp.lib import testing_utils
from lit_nlp.lib import validation


class ValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="all_required_all_present",
          spec={
              "score": types.Scalar(),
              "text": types.TextSegment(),
          },
          examples=[{"score": 0, "text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="some_required_all_present",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(),
          },
          examples=[{"score": 0, "text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="some_required_some_present",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(),
          },
          examples=[{"text": "a"}, {"text": "b"}],
      ),
  )
  def test_validate_dataset_with_base_dataset(
      self, spec: types.Spec, examples: list[types.JsonDict]
  ):
    ds = dataset.Dataset(spec=spec, examples=examples)
    try:
      validation.validate_dataset(ds, False)
    except ValueError:
      self.fail("Raised unexpected error.")

  @parameterized.named_parameters(
      dict(
          testcase_name="all_required_all_present",
          spec={
              "score": types.Scalar(),
              "text": types.TextSegment(),
          },
          examples=[
              {"data": {"score": 0, "text": "a"}, "id": 0, "meta": {}},
              {"data": {"score": 1, "text": "b"}, "id": 1, "meta": {}},
          ],
      ),
      dict(
          testcase_name="some_required_all_present",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(),
          },
          examples=[
              {"data": {"score": 0, "text": "a"}, "id": 0, "meta": {}},
              {"data": {"score": 1, "text": "b"}, "id": 1, "meta": {}},
          ],
      ),
      dict(
          testcase_name="some_required_some_present",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(),
          },
          examples=[
              {"data": {"text": "a"}, "id": 0, "meta": {}},
              {"data": {"text": "b"}, "id": 1, "meta": {}},
          ],
      ),
  )
  def test_validate_dataset_with_indexed_dataset(
      self, spec: types.Spec, examples: list[types.IndexedInput]
  ):
    ds = dataset.IndexedDataset(
        spec=spec, id_fn=lambda a: a, indexed_examples=examples
    )
    try:
      validation.validate_dataset(ds, False)
    except ValueError:
      self.fail("Raised unexpected error.")

  @parameterized.named_parameters(
      dict(
          testcase_name="first_field_not_required",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(),
          },
          examples=[{"score": 0, "text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="second_field_not_required",
          spec={
              "score": types.Scalar(),
              "text": types.TextSegment(required=False),
          },
          examples=[{"score": 0, "text": "a"}, {"score": 1, "text": "b"}],
      ),
  )
  def test_validate_dataset_raises_for_enforce_all_fields_required(
      self, spec: types.Spec, examples: list[types.JsonDict]
  ):
    ds = dataset.Dataset(spec=spec, examples=examples)
    with self.assertRaises(ValueError):
      validation.validate_dataset(ds, enforce_all_fields_required=True)

  @parameterized.named_parameters(
      dict(
          testcase_name="field_not_present_in_first_element",
          examples=[{"text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="field_not_present_in_second_element",
          examples=[{"score": 0, "text": "a"}, {"text": "b"}],
      ),
      dict(
          testcase_name="first_field_is_None_in_first_element",
          examples=[{"score": None, "text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="first_field_is_None_in_second_element",
          examples=[{"score": 0, "text": "a"}, {"score": None, "text": "b"}],
      ),
  )
  def test_validate_dataset_raises_for_required_fields(
      self, examples: list[types.JsonDict]
  ):
    spec = {"score": types.Scalar(), "text": types.TextSegment()}
    ds = dataset.Dataset(spec=spec, examples=examples)
    with self.assertRaises(ValueError):
      validation.validate_dataset(ds)

  @parameterized.named_parameters(
      dict(
          testcase_name="Scalar_is_str_in_first_element",
          examples=[{"score": "0", "text": "a"}, {"score": 1, "text": "b"}],
      ),
      dict(
          testcase_name="Scalar_is_str_in_second_element",
          examples=[{"score": 0, "text": "a"}, {"score": "1", "text": "b"}],
      ),
  )
  def test_validate_dataset_raises_for_malformed_fields(
      self, examples: list[types.JsonDict]
  ):
    spec = {"score": types.Scalar(), "text": types.TextSegment()}
    ds = dataset.Dataset(spec=spec, examples=examples)
    with self.assertRaises(ValueError):
      validation.validate_dataset(ds)

  @parameterized.named_parameters(
      dict(
          testcase_name="bad_everything_under_max_enforcement_logs_eight_times",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(required=False)
          },
          examples=[
              {},
              {"score": None, "text": None},
              {"score": "2", "text": True},
          ],
          enforce_all_fields_required=True,
          expected_log_count=8,
      ),
      dict(
          testcase_name="malformed_content_logs_twice",
          spec={"score": types.Scalar(), "text": types.TextSegment()},
          examples=[{"score": "0", "text": "a"}, {"score": 1, "text": True}],
          enforce_all_fields_required=False,
          expected_log_count=2,
      ),
      dict(
          testcase_name="missing_required_field_logs_twice",
          spec={"score": types.Scalar(), "text": types.TextSegment()},
          examples=[{"text": "a"}, {"score": 1}],
          enforce_all_fields_required=False,
          expected_log_count=2,
      ),
      dict(
          testcase_name="optional_fields_when_enforcing_required_logs_twice",
          spec={
              "score": types.Scalar(required=False),
              "text": types.TextSegment(required=False)
          },
          examples=[{"score": 0, "text": "a"}, {"score": 1, "text": "b"}],
          enforce_all_fields_required=True,
          expected_log_count=2,
      ),
  )
  def test_validate_dataset_report_all_log_counts(
      self,
      spec: types.Spec,
      examples: list[types.JsonDict],
      enforce_all_fields_required: bool,
      expected_log_count: int,
  ):
    ds = dataset.Dataset(spec=spec, examples=examples)
    with self.assertLogs(level="ERROR") as logs:
      with self.assertRaises(ValueError):
        validation.validate_dataset(
            ds,
            enforce_all_fields_required=enforce_all_fields_required,
            report_all=True,
        )
      self.assertLen(logs.output, expected_log_count)

  @parameterized.named_parameters(
      ("all_present", [{"res": 1, "grad": [1.0]}, {"res": 1, "grad": [1.0]}]),
      ("only_required", [{"res": 1}, {"res": 1}]),
  )
  def test_validate_model(self, results: list[types.JsonDict]):
    in_spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "res": types.RegressionScore(parent="score"),
        "grad": types.Gradients(required=False),
    }
    datapoints = [
        {"score": 0, "text": "a"},
        {"score": 1, "text": "b"},
    ]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.CustomOutputModelForTesting(
        in_spec, out_spec, results
    )
    try:
      validation.validate_model(model, ds)
    except ValueError:
      self.fail("Raised unexpected error.")

  @parameterized.named_parameters(
      ("required_output_is_missing", [{}]),
      ("required_output_is_None", [{"res": None}]),
      ("required_output_is_malformed", [{"res": "bad"}]),
      ("optional_output_is_malformed", [{"res": 1, "grad": "bad"}]),
  )
  def test_validate_model_raises(self, results: list[types.JsonDict]):
    in_spec: types.Spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec: types.Spec = {
        "res": types.RegressionScore(parent="score"),
        "grad": types.Gradients(required=False),
    }
    datapoints: list[types.JsonDict] = [{"score": 0, "text": "a"}]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.CustomOutputModelForTesting(
        in_spec, out_spec, results
    )
    with self.assertRaises(ValueError):
      validation.validate_model(model, ds)

  def test_validate_model_report_all_log_counts(self):
    in_spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "res": types.RegressionScore(parent="score"),
    }
    datapoints = [
        {"score": 0, "text": "a"},
        {"score": 1, "text": "b"},
    ]
    results = [{"res": None}, {"res": "bad"}]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.CustomOutputModelForTesting(
        in_spec, out_spec, results
    )
    with self.assertLogs(level="ERROR") as logs:
      with self.assertRaises(ValueError):
        validation.validate_model(model, ds, True)
      self.assertLen(logs.output, 2)


if __name__ == "__main__":
  absltest.main()
