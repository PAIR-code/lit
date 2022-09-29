"""Data to support runtime flags."""

import enum


@enum.unique
class ValidationMode(enum.Enum):
  """All the validation mode options."""
  OFF = 'off'  # Do not validate datasets and model outputs.
  FIRST = 'first'  # Validate the first datapoint.
  ALL = 'all'  # Validate all datapoints.
  SAMPLE = 'sample'  # Validate a sample of 5% of datapoints.
