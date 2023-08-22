"""Interpreter component for models that return their own salience."""
from typing import Optional, Union

from lit_nlp.api import components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils

JsonDict = types.JsonDict

# Salience types are for feature-wise salience, token-wise salience,
# sequence-based salience, or a string (base64 encoded image string) for image
# salience.
SalienceTypes = Union[dtypes.FeatureSalience, dtypes.TokenSalience,
                      dtypes.SequenceSalienceMap, str]

_SALIENCE_FIELD_TYPES = (
    types.FeatureSalience, types.ImageSalience, types.TokenSalience,
    types.SequenceSalience)


class ModelSalience(components.Interpreter):
  """Model-provided salience interpreter."""

  def __init__(self, models: dict[str, lit_model.Model]):
    # Populate saliency fields in meta spec based on saliency returned by
    # model output specs.
    self._spec = {}
    for model_name, model in models.items():
      fields = self.find_fields(model)
      for field in fields:
        self._spec[f'{model_name}:{field}'] = model.output_spec()[field]

  def find_fields(self, model: lit_model.Model) -> list[str]:
    return utils.find_spec_keys(model.output_spec(), _SALIENCE_FIELD_TYPES)

  def _run_single(self, ex: JsonDict, mo: JsonDict, fields: list[str],
                  model: lit_model.Model) -> dict[str, SalienceTypes]:
    # Extract the saliency outputs from the model.
    result = {}
    for sal_field in fields:
      result[sal_field] = mo[sal_field]
    return result

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    del dataset
    del config

    fields = self.find_fields(model)

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    return [
        self._run_single(ex, mo, fields, model)
        for ex, mo in zip(inputs, model_outputs)
    ]

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused as salience comes from the model.
    return utils.spec_contains(model.output_spec(), _SALIENCE_FIELD_TYPES)

  def meta_spec(self) -> types.Spec:
    return self._spec
