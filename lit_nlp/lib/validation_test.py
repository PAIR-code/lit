"""Tests for validation."""


from absl.testing import absltest
from lit_nlp.api import dataset
from lit_nlp.api import types
from lit_nlp.lib import testing_utils
from lit_nlp.lib import validation


class ValidationTest(absltest.TestCase):

  def test_validate_dataset(self):
    spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": 0,
            "text": "b"
        },
    ]
    ds = dataset.Dataset(spec, datapoints)
    try:
      validation.validate_dataset(ds, False)
    except ValueError:
      self.fail("Raised unexpected error.")

  def test_validate_dataset_fail_bad_scalar(self):
    spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    datapoints = [
        {
            "score": "bad",
            "text": "a"
        },
        {
            "score": 0,
            "text": "b"
        },
    ]
    ds = dataset.Dataset(spec, datapoints)
    self.assertRaises(ValueError, validation.validate_dataset, ds, False)
    self.assertRaises(ValueError, validation.validate_dataset, ds, True)

  def test_validate_dataset_validate_all(self):
    spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": "bad",
            "text": "b"
        },
    ]
    ds = dataset.Dataset(spec, datapoints)
    self.assertRaises(ValueError, validation.validate_dataset, ds, False)

  def test_validate_model(self):
    in_spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "res": types.RegressionScore(parent="score"),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": 1,
            "text": "b"
        },
    ]
    results = [{"res": 1}, {"res": 1}]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.TestCustomOutputModel(in_spec, out_spec, results)
    try:
      validation.validate_model(model, ds, True)
    except ValueError:
      self.fail("Raised unexpected error.")

  def test_validate_model_fail(self):
    in_spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "res": types.RegressionScore(parent="score"),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": 1,
            "text": "b"
        },
    ]
    results = [{"res": "bad"}, {"res": 1}]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.TestCustomOutputModel(in_spec, out_spec, results)
    self.assertRaises(
        ValueError, validation.validate_model, model, ds, False)

  def test_validate_model_validate_all(self):
    in_spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "res": types.RegressionScore(parent="score"),
    }
    datapoints = [
        {
            "score": 0,
            "text": "a"
        },
        {
            "score": 1,
            "text": "b"
        },
    ]
    results = [{"res": 1}, {"res": "bad"}]
    ds = dataset.Dataset(in_spec, datapoints)
    model = testing_utils.TestCustomOutputModel(in_spec, out_spec, results)
    self.assertRaises(
        ValueError, validation.validate_model, model, ds, False)


if __name__ == "__main__":
  absltest.main()
