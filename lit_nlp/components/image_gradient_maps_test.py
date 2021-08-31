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
# Lint as: python3
"""Tests for lit_nlp.components.gradient_maps."""

from typing import List
from absl.testing import absltest
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.api.model import JsonDict
from lit_nlp.components import image_gradient_maps
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image as PILImage


class ClassificationTestModel(lit_model.Model):

  LABELS = ['Dummy', 'Cat', 'Dog']
  GRADIENT_SHAPE = (60, 40, 3)

  def max_minibatch_size(self) -> int:
    return 10

  def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
    result = []
    for i, _ in enumerate(inputs):
      result.append({
          'preds': np.random.rand(len(self.LABELS)),
          'grads': np.random.rand(*self.GRADIENT_SHAPE),
          'grad_target': self.LABELS[i],
      })
    return result

  def input_spec(self):
    return {
        'image':
            lit_types.ImageBytes(),
        'grad_target':
            lit_types.CategoryLabel(vocab=self.LABELS, required=False)
    }

  def output_spec(self):
    return {
        'preds':
            lit_types.MulticlassPreds(vocab=self.LABELS),
        'grads':
            lit_types.ImageGradients(
                align='image', grad_target_field_key='grad_target'),
    }


class RegressionTestModel(lit_model.Model):
  """A test model for testing the regression case."""

  GRADIENT_SHAPE = (40, 20, 3)

  def max_minibatch_size(self) -> int:
    return 10

  def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
    """Simulates regression of x1 + 2 * x2 using elements of the image array."""
    result = []
    for example in inputs:
      img = example['image']
      if isinstance(img, str):
        img = image_utils.convert_image_str_to_array(
            img, shape=self.GRADIENT_SHAPE)
      pred = img[0, 0, 0] + img[1, 0, 0] * 2
      grad = np.zeros(shape=self.GRADIENT_SHAPE)
      grad[0, 0, 0] = 1
      grad[1, 0, 0] = 2
      result.append({'pred': pred, 'grads': grad})
    return result

  def input_spec(self):
    return {
        'image': lit_types.ImageBytes(),
    }

  def output_spec(self):
    return {
        'pred': lit_types.RegressionScore(),
        'grads': lit_types.ImageGradients(align='image'),
    }


class ImageGradientsMapsTest(absltest.TestCase):

  def test_interpreter(self):
    interpreter = image_gradient_maps.VanillaGradients()
    model = ClassificationTestModel()
    self.assertTrue(interpreter.is_compatible(model))

    pil_image = PILImage.new(mode='RGB', size=(300, 200))
    inp = {'image': image_utils.convert_pil_to_image_str(pil_image)}
    run_output = interpreter.run(
        inputs=[inp], model=model, dataset=lit_dataset.Dataset())[0]

    self.assertIn('grads', run_output)
    self.assertIsInstance(run_output['grads'], str)

  def test_regression(self):
    interpreter = image_gradient_maps.GuidedIG()
    model = RegressionTestModel()
    self.assertTrue(interpreter.is_compatible(model))

    input_image_array = np.zeros(shape=[20, 15, 3], dtype=np.uint8)
    input_image_array[0, 0, 0] = 10
    input_image_array[1, 0, 0] = 20

    pil_image = PILImage.fromarray(input_image_array, mode='RGB')

    inp = {'image': image_utils.convert_pil_to_image_str(pil_image)}
    run_output = interpreter.run(
        inputs=[inp], model=model, dataset=lit_dataset.Dataset())[0]
    self.assertIn('grads', run_output)
    overlay_str = run_output['grads']
    overlay_bytes = image_utils.convert_image_str_to_array(
        overlay_str, shape=RegressionTestModel.GRADIENT_SHAPE)
    self.assertIsNotNone(overlay_bytes)
    self.assertSequenceEqual(overlay_bytes.shape,
                             RegressionTestModel.GRADIENT_SHAPE)


if __name__ == '__main__':
  absltest.main()
