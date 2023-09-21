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
"""A collection of gradient based saliency interpreters for images.

This module implements interpreters that use the pair-code saliency library to
generate gradient based saliency maps.
"""

import abc
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, cast

from absl import logging
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.lib import image_utils
from lit_nlp.lib import utils as lit_utils
import numpy as np
import saliency


JsonDict = types.JsonDict
Spec = types.Spec

TARGET_INFO_KEY = '_salience_target'

# The key name for a configuration parameter that specifies the number
# of steps to use for integrated gradients approximation.
INTERPOLATION_KEY = 'Interpolation steps'
# The key name for a configuration parameter that specifies the maximum distance
# that the Guided IG path can deviate from the straight line.
MAX_DIST_KEY = 'Max. distance'

# Visualization constants.

# The color map to use for signed saliency methods such as Integrated
# Gradients.
DIVERGING_COLOR_MAP = 'bwr'
# The color map to use for unsigned saliency methods such as XRAI.
SEQUENTIAL_COLOR_MAP = 'summer'
# The amount of clipping to apply for saliency visualization. E.g., the value of
# 0.01 means that the top 1% of saliency values are clipped to match the 99%
# percentile.
CLIPPING_FRACTION = 0.01
# The value of alpha channel to apply to the saliency visualization layer that
# displays regions instead of pixels. The lower is the value the more visible is
# the original image underneath the saliency layer.
AREA_SALIENCY_ALPHA_MUL = 0.7
# The value of alpha channel to apply to the pixel-based saliency visualization
# layer that  The lower is the value the more visible is the original image
# underneath the saliency layer.
PIXEL_SALIENCY_ALPHA_MUL = 1.0

# Default value of IG steps.
IG_STEPS = 10

_SUPPORTED_PRED_TYPES = (types.MulticlassPreds, types.RegressionScore)


class SupportedFields(NamedTuple):
  """The collection of field names that are required to calculate saliency."""
  grad_field_key: str
  image_field_key: str
  grad_target_field_key: str
  preds_field_key: str


def find_supported_fields(input_spec: Spec,
                          output_spec: Spec) -> Optional[SupportedFields]:
  """Returns fields from the model specs that are needed for saliency ."""

  # Find all ImageGradients fields.
  grad_field_keys = lit_utils.find_spec_keys(output_spec, types.ImageGradients)
  # Models with more than one gradient field are not supported.
  if not grad_field_keys or len(grad_field_keys) != 1:
    logging.warning('Models must have exactly 1 ImageGradients field, found %i',
                    len(grad_field_keys))
    return None

  grad_field_key = grad_field_keys[0]
  grad_field_value = cast(types.ImageGradients, output_spec[grad_field_key])

  # Find image fields that correspond to grad_field.
  image_field_key = grad_field_value.align
  if not isinstance(input_spec.get(image_field_key), types.ImageBytes):
    logging.warning(
        'Could not find aligned ImageBytes field, %s, in input spec',
        str(grad_field_value.align))
    return None

  # Find gradient target fields in the input if it is a multiclass
  # classification model. The value of None means that it is a regression or
  # single class classification model.
  multiclass = grad_field_value.grad_target_field_key is not None
  if multiclass:
    grad_target_field_key = grad_field_value.grad_target_field_key
    if not isinstance(
        input_spec.get(grad_target_field_key), types.CategoryLabel):
      logging.warning(
          'Could not find compatible CategoryLabel field, %s, in input spec',
          str(grad_target_field_key))
      return None
  else:
    grad_target_field_key = None

  # Find prediction field names.
  preds_field_keys = lit_utils.find_spec_keys(
      output_spec, _SUPPORTED_PRED_TYPES
  )
  # Models with more than one prediction field are not supported.
  if not preds_field_keys or len(preds_field_keys) != 1:
    logging.warning('Models must have exactly 1 predicition field, found %i',
                    len(preds_field_keys))
    return None
  preds_field_key = preds_field_keys[0]

  return SupportedFields(
      grad_field_key=grad_field_key,
      image_field_key=image_field_key,
      grad_target_field_key=grad_target_field_key,
      preds_field_key=preds_field_key)

CallModelFunction = Callable[
    [np.ndarray, Any, list[str]], dict[str, np.ndarray]
]


def get_call_model_func(
    model: lit_model.Model,
    model_input: JsonDict,
    image_field_key: str,
    grad_field_key: str,
    grad_target_field_key: str,
    grad_target_label: str,
) -> CallModelFunction:
  """Returns a function that is used by the Saliency library to get gradients.

  Args:
    model: LIT model that is used to calculate actual gradients.
    model_input: the model input.
    image_field_key: the name (key) of the field in the model input that
      contains the image data with respect to which the gradients should be
      calculated.
    grad_field_key: the name (key) of the field in the model output that
      contains the computed gradients.
    grad_target_field_key: the name (key) of the field in the model input that
      is used to specify the label for which the gradients should be calculated.
      If the value is None then it is a regression or a single class
      classification model that has only one output.
    grad_target_label: the value of the label that should be passed as the
      `grad_target_field_name` value.

  Returns:
    The function that should be passed to the Saliency library.
  """

  def call_model_func(
      x_value_batch: np.ndarray, call_model_args, expected_keys: list[str]
  ) -> dict[str, np.ndarray]:
    """This function is called by the Saliency library to calculate gradients.

    Args:
      x_value_batch: a batch of inputs with respect to which the gradients
        should be calculated.
      call_model_args: unused.
      expected_keys: the list of expected keys that the return value should
        contain.

    Returns:
      A dictionary with gradients values for the batch.
    """
    del call_model_args  # Unused.

    # Iterate through the batch of saliency lib inputs and convert them to
    # a batch acceptable by the LIT model.
    model_inputs = []
    for x_value in x_value_batch:
      updates = {image_field_key: x_value}
      if grad_target_field_key is not None:
        updates[grad_target_field_key] = grad_target_label
      input_copy = lit_utils.make_modified_input(
          model_input, updates, 'ImageSalience'
      )
      model_inputs.append(input_copy)

    # Call the model to obtain gradients.
    predictions = model.predict(inputs=model_inputs)

    # Gradient results formatted for the saliency library.
    gradients_batch = [p[grad_field_key] for p in predictions]

    assert saliency.core.base.OUTPUT_LAYER_VALUES not in expected_keys

    # Convert the result to the format acceptable by the saliency library.
    return {
        saliency.core.base.INPUT_OUTPUT_GRADIENTS: np.asarray(gradients_batch),
    }

  return call_model_func


class SaliencyLibInterpreter(lit_components.Interpreter, metaclass=abc.ABCMeta):
  """A base class for all interpreters that use the Saliency library."""

  def run(
      self,
      inputs: list[JsonDict],
      model: lit_model.Model,
      dataset: lit_dataset.Dataset,
      model_outputs: Optional[list[JsonDict]] = None,
      config: Optional[JsonDict] = None,
  ) -> Optional[list[JsonDict]]:
    """Runs the component, given a model and input(s)."""
    input_spec = model.input_spec()
    output_spec = model.output_spec()
    config = config or {}

    if not inputs:
      return []

    # Find all fields required for the interpretation.
    supported_fields = find_supported_fields(input_spec, output_spec)
    if supported_fields is None:
      return None

    grad_field_key = supported_fields.grad_field_key
    image_field_key = supported_fields.image_field_key
    grad_target_field_key = supported_fields.grad_target_field_key

    if target_config := config.get(TARGET_INFO_KEY):
      preds_field_key = target_config['field']
    else:
      preds_field_key = supported_fields.preds_field_key

    preds_field_spec = output_spec[preds_field_key]
    if not isinstance(preds_field_spec, _SUPPORTED_PRED_TYPES):
      logging.warning(
          "Image Salience is not compatible with field '%s'", preds_field_key
      )
      return None

    # Determine the shape of gradients by calling the model with a single input
    # and extracting the shape from the gradient output.
    first_example_preds = list(model.predict([inputs[0]]))[0]
    grad_shape = first_example_preds[grad_field_key].shape

    # If it is a multiclass model, find the labels with respect to which we
    # should compute the gradients.
    if isinstance(preds_field_spec, types.MulticlassPreds):
      # Get class labels.
      label_vocab = list(preds_field_spec.vocab)

      if (target_config := config.get(TARGET_INFO_KEY)) and (
          target_class := target_config.get('index')
      ):
        grad_target_labels = [label_vocab[target_class] for _ in inputs]
      else:
        # Run the model in order to find the gradient target labels.
        outputs = list(model.predict(inputs))
        grad_target_labels = []
        for model_input, model_output in zip(inputs, outputs):
          if model_input.get(grad_target_field_key) is not None:
            grad_target_labels.append(model_input[grad_target_field_key])
          else:
            max_idx = int(np.argmax(model_output[preds_field_key]))
            grad_target_labels.append(label_vocab[max_idx])
    else:  # RegressionScore
      grad_target_labels = [None] * len(inputs)

    saliency_object = self.get_saliency_object()  # get this on class init?
    extra_saliency_params = self.get_extra_saliency_params(config)
    all_results = []
    for example, grad_target_label in zip(inputs, grad_target_labels):
      result = {}
      image_str = example[image_field_key]
      saliency_input = image_utils.convert_image_str_to_array(
          image_str=image_str, shape=grad_shape)
      call_model_func = get_call_model_func(
          model=model,
          model_input=example,
          image_field_key=image_field_key,
          grad_field_key=grad_field_key,
          grad_target_field_key=grad_target_field_key,
          grad_target_label=grad_target_label)
      attribution = self.make_saliency_call(
          saliency_object=saliency_object,
          x_value=saliency_input,
          call_model_function=call_model_func,
          extra_saliency_params=extra_saliency_params)
      if attribution.ndim == 3:
        attribution = attribution.sum(axis=2)
      viz_params = self.get_visualization_params()
      overlaid_image = image_utils.overlay_pixel_saliency(
          image_str, attribution, **viz_params)
      result[grad_field_key] = image_utils.convert_pil_to_image_str(
          overlaid_image)
      all_results.append(result)
    return all_results

  def is_compatible(self, model: lit_model.Model,
                    dataset: lit_dataset.Dataset) -> bool:
    del dataset  # Unused as salience comes from the model.
    fields = find_supported_fields(model.input_spec(), model.output_spec())
    return fields is not None

  def config_spec(self) -> types.Spec:
    return {'_salience_target': types.SalienceTargetInfo()}

  def meta_spec(self) -> types.Spec:
    return {'saliency': types.ImageSalience(autorun=False)}

  def make_saliency_call(
      self,
      saliency_object: saliency.core.CoreSaliency,
      x_value: np.ndarray,
      call_model_function: CallModelFunction,
      extra_saliency_params: dict[str, Any],
  ) -> np.ndarray:
    return saliency_object.GetMask(
        x_value=x_value,
        call_model_function=call_model_function,
        **extra_saliency_params)

  @abc.abstractmethod
  def get_saliency_object(self):
    """Returns a saliency library instance to be used for the explanation."""
    pass

  @abc.abstractmethod
  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    """Returns extra parameters to be passed to the GetMask() method."""
    pass

  @abc.abstractmethod
  def get_visualization_params(self) -> dict[str, Any]:
    """Returns visualization parameters."""
    pass


class VanillaGradients(SaliencyLibInterpreter):
  """Vanilla gradients interpreter."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.GradientSaliency()

  def get_extra_saliency_params(self, unused_config) -> dict[str, Any]:
    return {}

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': DIVERGING_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': PIXEL_SALIENCY_ALPHA_MUL,
        'signed': True,
        'pixel_saliency': True
    }


class IntegratedGradients(SaliencyLibInterpreter):
  """Integrated Gradients interpreter."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.IntegratedGradients()

  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    return {'x_steps': int(config.get(INTERPOLATION_KEY, IG_STEPS))}

  def config_spec(self) -> types.Spec:
    return super().config_spec() | {
        INTERPOLATION_KEY: types.Scalar(
            min_val=5, max_val=200, default=IG_STEPS, step=1, required=False
        )
    }

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': DIVERGING_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': PIXEL_SALIENCY_ALPHA_MUL,
        'signed': True,
        'pixel_saliency': True
    }


class BlurIG(SaliencyLibInterpreter):
  """Blur IG interpreter."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.BlurIG()

  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    return {'steps': int(config.get(INTERPOLATION_KEY, IG_STEPS))}

  def config_spec(self) -> types.Spec:
    return super().config_spec() | {
        INTERPOLATION_KEY: types.Scalar(
            min_val=5, max_val=200, default=IG_STEPS, step=1, required=False
        )
    }

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': DIVERGING_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': PIXEL_SALIENCY_ALPHA_MUL,
        'signed': True,
        'pixel_saliency': True
    }


class GuidedIG(SaliencyLibInterpreter):
  """Guided Integrated Gradients interpreter."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.GuidedIG()

  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    return {
        'x_steps': int(config.get(INTERPOLATION_KEY, IG_STEPS)),
        'max_dist': float(config.get(MAX_DIST_KEY, 0.1)),
        'fraction': 0.25,
    }

  def config_spec(self) -> types.Spec:
    return super().config_spec() | {
        INTERPOLATION_KEY: types.Scalar(
            min_val=5, max_val=200, default=IG_STEPS, step=1, required=False
        ),
        MAX_DIST_KEY: types.Scalar(
            min_val=0.0, max_val=1.0, default=0.1, step=0.02, required=False
        ),
    }

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': DIVERGING_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': PIXEL_SALIENCY_ALPHA_MUL,
        'signed': True,
        'pixel_saliency': True
    }


class XRAI(SaliencyLibInterpreter):
  """XRAI Interpreter."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.XRAI()

  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    return {'steps': int(config.get(INTERPOLATION_KEY, IG_STEPS))}

  def config_spec(self) -> types.Spec:
    return super().config_spec() | {
        INTERPOLATION_KEY: types.Scalar(
            min_val=5, max_val=200, default=IG_STEPS, step=1, required=False
        )
    }

  def make_saliency_call(
      self,
      saliency_object: saliency.core.CoreSaliency,
      x_value: np.ndarray,
      call_model_function: CallModelFunction,
      extra_saliency_params: dict[str, Any],
  ) -> np.ndarray:
    xrai_params = saliency.core.XRAIParameters(
        steps=extra_saliency_params['steps'], algorithm='fast'
    )
    xrai_output = cast(saliency.core.XRAI, saliency_object).GetMaskWithDetails(
        x_value=x_value,
        call_model_function=call_model_function,
        extra_parameters=xrai_params)
    return xrai_output.attribution_mask

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': SEQUENTIAL_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': AREA_SALIENCY_ALPHA_MUL,
        'signed': False,
        'pixel_saliency': False
    }


class XRAIGIG(SaliencyLibInterpreter):
  """XRAI Interpreter that uses Guided IG as the base attribution."""

  def get_saliency_object(self) -> saliency.core.CoreSaliency:
    return saliency.core.XRAI()

  def get_extra_saliency_params(self, config: JsonDict) -> dict[str, Any]:
    return {
        'x_steps': int(config.get(INTERPOLATION_KEY, IG_STEPS)),
        'max_dist': float(config.get(MAX_DIST_KEY, 0.1)),
        'fraction': 0.25,
    }

  def config_spec(self) -> types.Spec:
    return super().config_spec() | {
        INTERPOLATION_KEY: types.Scalar(
            min_val=5, max_val=200, default=IG_STEPS, step=1, required=False
        ),
        MAX_DIST_KEY: types.Scalar(
            min_val=0.0, max_val=1.0, default=0.1, step=0.02, required=False
        ),
    }

  def make_saliency_call(
      self,
      saliency_object: saliency.core.CoreSaliency,
      x_value: np.ndarray,
      call_model_function: CallModelFunction,
      extra_saliency_params: dict[str, Any],
  ) -> np.ndarray:
    gig_object = saliency.core.GuidedIG()
    gig_saliency = gig_object.GetMask(
        x_value=x_value,
        call_model_function=call_model_function,
        **extra_saliency_params)

    xrai_params = saliency.core.XRAIParameters(
        steps=extra_saliency_params['x_steps'], algorithm='fast')
    xrai_output = cast(saliency.core.XRAI, saliency_object).GetMaskWithDetails(
        x_value=x_value,
        call_model_function=call_model_function,
        base_attribution=gig_saliency,
        extra_parameters=xrai_params)
    return xrai_output.attribution_mask

  def get_visualization_params(self) -> dict[str, Any]:
    return {
        'cm_name': SEQUENTIAL_COLOR_MAP,
        'clip_fraction': CLIPPING_FRACTION,
        'alpha_mul': AREA_SALIENCY_ALPHA_MUL,
        'signed': False,
        'pixel_saliency': False
    }


def all_interpreters():
  return {
      'Grad': VanillaGradients(),
      'Integrated Gradients': IntegratedGradients(),
      'Blur IG': BlurIG(),
      'Guided IG': GuidedIG(),
      'XRAI': XRAI(),
      'XRAI GIG': XRAIGIG(),
  }
