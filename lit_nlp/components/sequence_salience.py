"""Interpreter components for seq2seq salience."""
from typing import Optional

import Levenshtein  # TEMPORARY; for dummy salience
from lit_nlp.api import components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import utils
import numpy as np  # TEMPORARY; for dummy salience

JsonDict = types.JsonDict

_SUPPORTED_OUTPUTS = (types.GeneratedText, types.GeneratedTextCandidates)


class DummySequenceSalience(components.Interpreter):
  """Dummy-valued seq2seq salience, for testing."""

  def find_fields(self, model: lit_model.Model) -> dict[str, list[str]]:
    src_fields = utils.find_spec_keys(model.input_spec(), types.TextSegment)
    gen_fields = utils.find_spec_keys(model.output_spec(), _SUPPORTED_OUTPUTS)
    return {f: src_fields for f in gen_fields}

  def _dummy_sequence_salience(
      self, source_tokens: list[str], target_tokens: list[str]):
    """Compute salience matrix based on Levenshtein similarity."""
    all_input_tokens = source_tokens + target_tokens
    smat = np.zeros([len(target_tokens), len(all_input_tokens)])
    for i, out_tok in enumerate(target_tokens):
      for j, in_tok in enumerate(all_input_tokens[:len(source_tokens) + i]):
        smat[i, j] = 1.0 / (Levenshtein.distance(out_tok, in_tok) + 1.0)
    return smat

  def _run_single(
      self, ex: JsonDict, mo: JsonDict,
      field_map: dict[str, str]) -> dict[str, dtypes.SequenceSalienceMap]:
    result = {}  # dict[target_field name -> interpretations]
    for (target_field, source_fields) in field_map.items():
      source_tokens = []
      for sf in source_fields:
        source_tokens.extend(ex.get(sf, '').split())

      target = mo[target_field]
      if isinstance(target, list):
        target = target[0][0]  # text for first candidate
      target_tokens = target.split()

      salience = self._dummy_sequence_salience(source_tokens, target_tokens)
      result[target_field] = dtypes.SequenceSalienceMap(source_tokens,
                                                        target_tokens, salience)
    return result

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    del dataset
    del config

    field_map = self.find_fields(model)

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    return [
        self._run_single(ex, mo, field_map)
        for ex, mo in zip(inputs, model_outputs)
    ]

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused as salience comes from the model.
    has_inputs = utils.spec_contains(model.input_spec(), types.TextSegment)
    has_outputs = utils.spec_contains(model.output_spec(), _SUPPORTED_OUTPUTS)
    return has_inputs and has_outputs

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.SequenceSalience(autorun=True, signed=False)}
