"""Validators for datasets and models."""


from typing import cast
from absl import logging
from lit_nlp.api import dataset
from lit_nlp.api import model
from lit_nlp.api import types


def validate_dataset(ds: dataset.Dataset, report_all: bool):
  """Validate dataset entries against spec."""
  last_error = None
  for ex in ds.examples:
    for (key, entry) in ds.spec().items():
      if key not in ex or ex[key] is None:
        if entry.required:
          raise ValueError(
              f'Required dataset feature {key} missing from datapoint')
        else:
          continue
      try:
        entry.validate_input(ex[key], ds.spec(), cast(types.Input, ex))
      except ValueError as e:
        logging.exception('Failed validating input key %s', key)
        if report_all:
          last_error = e
        else:
          raise e
  if last_error:
    raise last_error


def validate_model(mod: model.Model, ds: dataset.Dataset, report_all: bool):
  """Validate model usage on dataset against specs."""
  last_error = None
  outputs = list(mod.predict(ds.examples))
  for ex, output in zip(ds.examples, outputs):
    for (key, entry) in mod.output_spec().items():
      if key not in output or output[key] is None:
        if entry.required:
          raise ValueError(
              f'Required model output {key} missing from prediction result')
        else:
          continue
      try:
        entry.validate_output(
            output[key], mod.output_spec(), output, mod.input_spec(), ds.spec(),
            cast(types.Input, ex))
      except ValueError as e:
        logging.exception('Failed validating model output key %s', key)
        if report_all:
          last_error = e
        else:
          raise e
  if last_error:
    raise last_error
