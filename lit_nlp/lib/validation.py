"""Validators for datasets and models."""

from typing import cast, Optional
from absl import logging
from lit_nlp.api import dataset
from lit_nlp.api import model
from lit_nlp.api import types
import termcolor


def validate_dataset(
    ds: dataset.Dataset,
    enforce_all_fields_required: bool = False,
    report_all: bool = False,
) -> None:
  """Validate dataset entries against spec.

  Args:
    ds: The Dataset being validated.
    enforce_all_fields_required: If `True`, require that every field in the
      Dataset referenced by `ds` has `required=True`.
    report_all: If `True`, log all errors before raising the first error
      encountered in the validation of the Dataset referenced by `ds`.

  Raises:
    ValueError: The first instance of one of the following conditions occurring
      during validation:
      * A field in the Dataset's Spec has `required=False` when enforcing all
        fields are required.
      * The value for a field in an example is `None` when that field is
        required (either explicitly or when enforcing all fields are required).
      * The value for a field fails valdiation via `LitType.validate_input()`.
  """
  # If report_all is True, first_error stores the first error encountered during
  # the validation process, which is then raised at the end of processing.
  first_error: Optional[ValueError] = None
  # If report_all is True, first_error_origin stores the ValueError raised by
  # LitType.validate_input() if a datapoint fails validation.
  first_error_origin: Optional[ValueError] = None

  def raise_or_log_error(
      msg: str, origin: Optional[ValueError] = None
  ) -> ValueError:
    """Raise (if report_all=False) or log (and return) a validation error."""
    if report_all:
      logging.error(termcolor.colored(msg, 'red'))
      return ValueError(msg)
    else:
      raise ValueError(msg) from origin

  for key, entry in ds.spec().items():
    if enforce_all_fields_required and not entry.required:
      err = raise_or_log_error(
          f'Encountered a field, "{key}", that has required=False while'
          ' enforcing that all fields in the Dataset.spec must be requred.'
      )
      first_error = first_error or err

    for example in ds.examples:
      value = example.get(key)
      if value is None:
        if enforce_all_fields_required or entry.required:
          err = raise_or_log_error(
              f'Required dataset feature "{key}" missing from datapoint.'
          )
          first_error = first_error or err
      else:
        try:
          entry.validate_input(value, ds.spec(), cast(types.Input, example))
        except ValueError as e:
          err = raise_or_log_error(
              f'Failed while validating dataset field "{key}" of type'
              f' "{type(entry)}".',
              e,
          )
          first_error = first_error or err

  if first_error:
    raise first_error from first_error_origin


def validate_model(
    mod: model.Model, ds: dataset.Dataset, report_all: bool = False
) -> None:
  """Validate model usage on dataset against specs.

  Args:
    mod: The Model providing the predictions that are being validated.
    ds: The Dataset providing the examples for which the model referenced by
      `mod` makes predictions.
    report_all: If `True`, log all errors before raising the first error
      encountered in the validation of the Model referenced by `mod`.

  Raises:
    ValueError: The first instance of one of the following conditions occurring
      during validation:
      * A required output field is missing from a prediction.
      * The value for a predicted field fails valdiation via
        `LitType.validate_output()`.
  """
  # If report_all is True, first_error stores the first error encountered during
  # the validation process, which is then raised at the end of processing.
  first_error: Optional[ValueError] = None
  # If report_all is True, first_error_origin stores the ValueError raised by
  # LitType.validate_output() if a datapoint fails validation.
  first_error_origin: Optional[ValueError] = None

  def raise_or_log_error(
      msg: str, origin: Optional[ValueError] = None
  ) -> ValueError:
    """Raise (if report_all=False) or log (and return) a validation error."""
    if report_all:
      logging.error(termcolor.colored(msg, 'red'))
      return ValueError(msg)
    else:
      raise ValueError(msg) from origin

  outputs = list(mod.predict(ds.examples))
  for ex, output in zip(ds.examples, outputs):
    for (key, entry) in mod.output_spec().items():
      value = output.get(key)
      if value is None:
        if entry.required:
          err = raise_or_log_error(
              f'Required model output "{key}" is missing from prediction.'
          )
          first_error = first_error or err
      else:
        try:
          entry.validate_output(
              value,
              mod.output_spec(),
              output,
              mod.input_spec(),
              ds.spec(),
              cast(types.Input, ex),
          )
        except ValueError as e:
          err = raise_or_log_error(
              f'Failed while validating model output field "{key}" of type'
              f' "{type(entry)}".',
              e,
          )
          first_error = first_error or err

  if first_error:
    raise first_error from first_error_origin
