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
"""Gradient-based attribution."""

from typing import cast, Optional

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components.citrus import utils as citrus_utils
from lit_nlp.lib import utils
import numpy as np


JsonDict = types.JsonDict
Spec = types.Spec

CLASS_KEY = 'Class to explain'
NORMALIZATION_KEY = 'Normalize'
INTERPOLATION_KEY = 'Interpolation steps'


class GradientNorm(lit_components.Interpreter):
  """Salience map from gradient L2 norm."""

  def find_fields(self, output_spec: Spec) -> list[str]:
    # Find TokenGradients fields
    supported_fields: list[str] = []

    # Check that these are aligned to Tokens fields
    for f in utils.find_spec_keys(output_spec, types.TokenGradients):
      tokens_field = cast(types.TokenGradients, output_spec[f]).align
      is_valid_tokens = (
          tokens_field is not None and tokens_field in output_spec and
          isinstance(output_spec[tokens_field], types.Tokens))
      if not is_valid_tokens:
        logging.info('Skipping %s. Invalid tokens field, %s', str(f),
                     str(tokens_field))
        continue
      supported_fields.append(f)
    return supported_fields

  def _interpret(self, grads: np.ndarray, tokens: np.ndarray):
    assert grads.shape[0] == len(tokens)
    # Norm of dy/d(embs)
    grad_norm = np.linalg.norm(grads, axis=1)
    grad_norm /= np.sum(grad_norm)
    # <float32>[num_tokens]
    return grad_norm

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Run this component, given a model and input(s)."""
    del dataset, config
    # Find gradient fields to interpret
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info('Found fields for gradient attribution: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    all_results = []
    for o in model_outputs:
      # Dict[field name -> interpretations]
      result = {}
      for grad_field in grad_fields:
        token_field = cast(types.TokenGradients, output_spec[grad_field]).align
        tokens = o[token_field]
        scores = self._interpret(o[grad_field], tokens)
        result[grad_field] = dtypes.TokenSalience(tokens, scores)
      all_results.append(result)

    return all_results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused by Grad L2 Norm
    return bool(self.find_fields(model.output_spec()))

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.TokenSalience(autorun=True, signed=False)}


class GradientDotInput(lit_components.Interpreter):
  """Salience map using the values of gradient * input as attribution."""

  def find_fields(self, output_spec: Spec) -> list[str]:
    # Find and check that TokenGradients fields are aligned to Tokens fields
    aligned_fields = []
    for f in utils.find_spec_keys(output_spec, types.TokenGradients):
      field_spec = cast(types.TokenGradients, output_spec[f])
      tokens_field = field_spec.align
      is_valid_tokens = (
          tokens_field is not None and tokens_field in output_spec and
          isinstance(output_spec[tokens_field], types.Tokens))
      if not is_valid_tokens:
        logging.info('Skipping %s. Invalid tokens field, %s', str(f),
                     str(tokens_field))
        continue

      embeddings_field = field_spec.grad_for
      is_valid_embeddings = (
          embeddings_field is not None and embeddings_field in output_spec and
          isinstance(output_spec[embeddings_field], types.TokenEmbeddings))
      if not is_valid_embeddings:
        logging.info('Skipping %s. Invalid emebeddings field, %s.', str(f),
                     str(tokens_field))
        continue

      aligned_fields.append(f)
    return aligned_fields

  def _interpret(self, grads: np.ndarray, embs: np.ndarray):
    assert grads.shape == embs.shape

    # dot product of gradients and embeddings
    # <float32>[num_tokens]
    grad_dot_input = np.sum(grads * embs, axis=-1)
    scores = citrus_utils.normalize_scores(grad_dot_input)
    return scores

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Run this component, given a model and input(s)."""
    # Find gradient fields to interpret
    output_spec = model.output_spec()
    grad_fields = self.find_fields(output_spec)
    logging.info('Found fields for gradient attribution: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))
    assert len(model_outputs) == len(inputs)

    all_results = []
    for o in model_outputs:
      # Dict[field name -> interpretations]
      result = {}
      for grad_field in grad_fields:
        embeddings_field = cast(types.TokenGradients,
                                output_spec[grad_field]).grad_for
        scores = self._interpret(o[grad_field], o[embeddings_field])

        token_field = cast(types.TokenGradients, output_spec[grad_field]).align
        tokens = o[token_field]
        result[grad_field] = dtypes.TokenSalience(tokens, scores)
      all_results.append(result)

    return all_results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused by Grad*Input
    return bool(self.find_fields(model.output_spec()))

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.TokenSalience(autorun=True, signed=True)}


class IntegratedGradients(lit_components.Interpreter):
  """Salience map from Integrated Gradients.

  Integrated Gradients is an attribution method originally proposed in
  Sundararajan et al. (https://arxiv.org/abs/1703.01365), which attributes an
  importance value for each input feature based on the gradients of the model
  output with respect to the input. The feature attribution values are
  calculated by taking the integral of gradients along a straight path from a
  baseline to the input being analyzed. The original implementation can be
  found at: https://github.com/ankurtaly/Integrated-Gradients/blob/master/
  BertModel/bert_model_utils.py

  This component requires that the following fields in the model spec. Field
  names like `embs` are placeholders; you can call them whatever you like,
  and as with other LIT components multiple segments are supported.
    Output:
      - TokenEmbeddings (`embs`) to return the input embeddings
      - TokenGradients (`grads`) to return gradients w.r.t. `embs`
      - A label field (`target`) to return the label that `grads`
        was computed for. This is usually a CategoryLabel, but can be anything
        since it will just be fed back into the model.
    Input
      - TokenEmbeddings (`embs`) to accept the modified input embeddings
      - A label field to (`target`) to pin the gradient target to the same
        label for all integral steps, since the argmax prediction may change.
  """

  def __init__(self,
               autorun: bool = False,
               class_key: str = '',
               interpolation_steps: int = 30,
               normalize: bool = True):
    """Cretaes an IntegratedGradients interpreter.

    Args:
      autorun: Determines if this intepreter should run automatically.
      class_key: The class to explain.
      interpolation_steps: The number of steps to interpolate.
      normalize: Flag to enable/disable normalization.
    """
    self._autorun: bool = autorun
    self._class_key: str = class_key
    self._interpolation_steps: int = interpolation_steps
    self._normalize: bool = normalize

  def find_fields(self, input_spec: Spec, output_spec: Spec) -> list[str]:
    # Find and check that TokenGradients fields are aligned to Tokens fields
    aligned_fields = []
    for f in utils.find_spec_keys(output_spec, types.TokenGradients):
      field_spec = cast(types.TokenGradients, output_spec[f])
      tokens_field = field_spec.align
      embeddings_field = field_spec.grad_for
      grad_key = field_spec.grad_target_field_key

      if not isinstance(output_spec.get(tokens_field), types.Tokens):
        logging.info('Skipping %s. Invalid tokens field, %s.', str(f),
                     str(tokens_field))
        continue

      is_embs_valid = (
          isinstance(input_spec.get(embeddings_field),
                     types.TokenEmbeddings) and
          isinstance(output_spec.get(embeddings_field), types.TokenEmbeddings))
      if not is_embs_valid:
        logging.info('Skipping %s. Invalid embeddings field, %s.', str(f),
                     str(tokens_field))
        continue

      is_grad_cls_valid = grad_key in input_spec and grad_key in output_spec
      if not is_grad_cls_valid:
        logging.info('Skipping %s. Invalid gradient class field, %s.', str(f),
                     str(tokens_field))
        continue

      aligned_fields.append(f)
    return aligned_fields

  def get_interpolated_inputs(self, baseline: np.ndarray, target: np.ndarray,
                              num_steps: int) -> np.ndarray:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    if num_steps <= 0: return np.array([])
    if num_steps == 1: return np.array([baseline, target])

    delta = target - baseline  # <float32>[num_tokens, emb_size]
    # Creates scale values array of shape [num_steps, num_tokens, emb_dim],
    # where the values in scales[i] are the ith step from np.linspace.
    # <float32>[num_steps, 1, 1]
    scales = np.linspace(0, 1, num_steps + 1,
                         dtype=np.float32)[:, np.newaxis, np.newaxis]

    shape = (num_steps + 1,) + delta.shape
    # <float32>[num_steps, num_tokens, emb_size]
    deltas = scales * np.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs  # <float32>[num_steps, num_tokens, emb_size]

  def estimate_integral(self, path_gradients: np.ndarray) -> np.ndarray:
    """Estimates the integral of the path_gradients using trapezoid rule."""

    path_gradients = (path_gradients[:-1] + path_gradients[1:]) / 2

    # There are num_steps elements in the path_gradients. Summing num_steps - 1
    # terms and dividing by num_steps - 1 is equivalent to taking
    # the average.
    return np.average(path_gradients, axis=0)

  def get_baseline(self, embeddings: np.ndarray) -> np.ndarray:
    """Returns baseline embeddings to use in Integrated Gradients."""

    # Replaces embeddings in the original input with the zero embedding, or
    # with the specified token embedding.
    baseline = np.zeros_like(embeddings)

    # TODO(ellenj): Add option to use a token's embedding as the baseline.
    return baseline

  def get_salience_result(self, model_input: JsonDict, model: lit_model.Model,
                          interpolation_steps: int, normalize: bool,
                          class_to_explain: str, model_output: JsonDict,
                          grad_fields: list[str]):
    result = {}

    output_spec = model.output_spec()
    # We ensure that the embedding and gradient class fields are present in the
    # model's input spec in find_fields().
    embeddings_fields = [
        cast(types.TokenGradients,
             output_spec[grad_field]).grad_for for grad_field in grad_fields]

    # The gradient class input is used to specify the target class of the
    # gradient calculation (if unspecified, this option defaults to the argmax,
    # which could flip between interpolated inputs).
    # If class_to_explain is emptystring, then explain the argmax class.
    grad_class_key = cast(types.TokenGradients,
                          output_spec[grad_fields[0]]).grad_target_field_key
    if class_to_explain == '':  # pylint: disable=g-explicit-bool-comparison
      grad_class = model_output[grad_class_key]
    else:
      grad_class = class_to_explain

    interpolated_inputs = {}
    all_embeddings = []
    all_baselines = []
    for embed_field in embeddings_fields:
      # <float32>[num_tokens, emb_size]
      embeddings = np.array(model_output[embed_field])
      all_embeddings.append(embeddings)

      # Starts with baseline of zeros. <float32>[num_tokens, emb_size]
      baseline = self.get_baseline(embeddings)
      all_baselines.append(baseline)

      # Get interpolated inputs from baseline to original embedding.
      # <float32>[interpolation_steps, num_tokens, emb_size]
      interpolated_inputs[embed_field] = self.get_interpolated_inputs(
          baseline, embeddings, interpolation_steps)

    # Create model inputs and populate embedding field(s).
    inputs_with_embeds = []
    for i in range(interpolation_steps):
      input_copy = dict(model_input)
      # Interpolates embeddings for all inputs simultaneously.
      for embed_field in embeddings_fields:
        # <float32>[num_tokens, emb_size]
        input_copy[embed_field] = interpolated_inputs[embed_field][i]
        input_copy[grad_class_key] = grad_class

      inputs_with_embeds.append(input_copy)
    embed_outputs = model.predict(inputs_with_embeds)

    # Create list with concatenated gradients for each interpolate input.
    gradients = []
    for o in embed_outputs:
      # <float32>[total_num_tokens, emb_size]
      interp_gradients = np.concatenate([o[field] for field in grad_fields])
      gradients.append(interp_gradients)
    # <float32>[interpolation_steps, total_num_tokens, emb_size]
    path_gradients = np.stack(gradients, axis=0)

    # Calculate integral
    # <float32>[total_num_tokens, emb_size]
    integral = self.estimate_integral(path_gradients)

    # <float32>[total_num_tokens, emb_size]
    concat_embeddings = np.concatenate(all_embeddings)

    # <float32>[total_num_tokens, emb_size]
    concat_baseline = np.concatenate(all_baselines)

    # <float32>[total_num_tokens, emb_size]
    integrated_gradients = integral * (np.array(concat_embeddings) -
                                       np.array(concat_baseline))
    # Dot product of integral values and (embeddings - baseline).
    # <float32>[total_num_tokens]
    attributions = np.sum(integrated_gradients, axis=-1)

    # <float32>[total_num_tokens]
    scores = citrus_utils.normalize_scores(
        attributions) if normalize else attributions

    for grad_field in grad_fields:
      # Format as salience map result.
      token_field = cast(types.TokenGradients, output_spec[grad_field]).align
      tokens = model_output[token_field]

      # Only use the scores that correspond to the tokens in this grad_field.
      # The gradients for all input embeddings were concatenated in the order
      # of the grad fields, so they can be sliced out in the same order.
      sliced_scores = scores[:len(tokens)]  # <float32>[num_tokens in field]
      scores = scores[len(tokens):]  # <float32>[num_remaining_tokens]

      assert len(tokens) == len(sliced_scores)
      result[grad_field] = dtypes.TokenSalience(tokens, sliced_scores)
    return result

  def run(self,
          inputs: list[JsonDict],
          model: lit_model.Model,
          dataset: lit_dataset.Dataset,
          model_outputs: Optional[list[JsonDict]] = None,
          config: Optional[JsonDict] = None) -> Optional[list[JsonDict]]:
    """Run this component, given a model and input(s)."""
    config = config or {}
    class_to_explain = config.get(CLASS_KEY, self._class_key)

    try:
      interpolation_steps = int(config.get(INTERPOLATION_KEY,
                                           self._interpolation_steps))
    except ValueError as parse_error:
      raise RuntimeError(
          'Failed to parse interpolation steps'
          f'from "{config[INTERPOLATION_KEY]}".') from parse_error

    normalization = config.get(NORMALIZATION_KEY, self._normalize)

    # Find gradient fields to interpret
    input_spec = model.input_spec()
    output_spec = model.output_spec()
    grad_fields = self.find_fields(input_spec, output_spec)
    logging.info('Found fields for integrated gradients: %s', str(grad_fields))
    if len(grad_fields) == 0:  # pylint: disable=g-explicit-length-test
      return None

    # Run model, if needed.
    if model_outputs is None:
      model_outputs = list(model.predict(inputs))

    all_results = []
    for model_output, model_input in zip(model_outputs, inputs):
      result = self.get_salience_result(model_input, model, interpolation_steps,
                                        normalization, class_to_explain,
                                        model_output, grad_fields)
      all_results.append(result)
    return all_results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused by IG
    return bool(self.find_fields(model.input_spec(), model.output_spec()))

  def config_spec(self) -> types.Spec:
    return {
        CLASS_KEY: types.TextSegment(default=self._class_key),
        NORMALIZATION_KEY: types.Boolean(default=self._normalize),
        INTERPOLATION_KEY:
            types.Scalar(
                min_val=5,
                max_val=100,
                default=self._interpolation_steps,
                step=1)
    }

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.TokenSalience(autorun=self._autorun, signed=True)}
