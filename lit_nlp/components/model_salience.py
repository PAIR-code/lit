"""Interpreter component for models that return their own salience."""
from typing import Dict, List, Optional, Union

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


class ModelSalience(components.Interpreter):
  """Model-provided salience interpreter."""

  def __init__(self, models: Dict[str, lit_model.Model]):
    # Populate saliency fields in meta spec based on saliency returned by
    # model output specs.
    self._spec = {}
    for model_name, model in models.items():
      fields = self.find_fields(model)
      for field in fields:
        self._spec[f'{model_name}:{field}'] = model.output_spec()[field]

  def find_fields(self, model: lit_model.Model) -> List[str]:
    sal_keys = utils.find_spec_keys(
        model.output_spec(),
        (types.FeatureSalience, types.ImageSalience, types.TokenSalience,
         types.SequenceSalience))
    return sal_keys

  def _run_single(
      self, ex: JsonDict, mo: JsonDict,
      fields: List[str], model: lit_model.Model) -> Dict[str, SalienceTypes]:
    # Extract the saliency outputs from the model.
    result = {}
    for sal_field in fields:
      result[sal_field] = mo[sal_field]
    return result

  def run(self,
          inputs: List[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[List[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[List[JsonDict]]:
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

  def is_compatible(self, model: lit_model.Model):
    return len(self.find_fields(model))

  def meta_spec(self) -> types.Spec:
    return self._spec
