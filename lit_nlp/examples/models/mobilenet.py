"""MobileNet model trained on ImageNet dataset."""

from typing import List

from lit_nlp.api import model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import imagenet_labels
from lit_nlp.lib import image_utils
from lit_nlp.lib import utils as lit_utils
import numpy as np
import tensorflow as tf

# Internal shape of the model input (h, w, c).
IMAGE_SHAPE = (224, 224, 3)


class MobileNet(model.Model):
  """MobileNet model trained on ImageNet dataset."""

  class MobileNetSpec(model.ModelSpec):

    def is_compatible_with_dataset(self, dataset_spec: lit_types.Spec) -> bool:
      image_field_names = lit_utils.find_spec_keys(dataset_spec,
                                                   lit_types.ImageBytes)
      return bool(image_field_names)

  def __init__(self) -> None:
    # Initialize imagenet labels.
    self.labels = [''] * len(imagenet_labels.IMAGENET_2012_LABELS)
    self.label_to_idx = {}
    for i, l in imagenet_labels.IMAGENET_2012_LABELS.items():
      l = l.split(',', 1)[0]
      self.labels[i] = l
      self.label_to_idx[l] = i

    self.model = tf.keras.applications.mobilenet_v2.MobileNetV2()

  def predict_minibatch(
      self, input_batch: List[lit_types.JsonDict]) -> List[lit_types.JsonDict]:
    output = []
    for example in input_batch:
      # Convert input to the model acceptable format.
      img_data = example['image']
      if isinstance(img_data, str):
        img_data = image_utils.convert_image_str_to_array(img_data, IMAGE_SHAPE)
      # Get predictions.
      x = img_data[np.newaxis, ...]
      x = tf.convert_to_tensor(x)
      preds = self.model(x).numpy()[0]
      # Determine the gradient target.
      grad_target = example.get('grad_target')
      if grad_target is None:
        grad_target_idx = np.argmax(preds)
      else:
        grad_target_idx = self.label_to_idx[grad_target]
      # Calculate gradients.
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = self.model(x)[0, grad_target_idx]
        grads = tape.gradient(y, x).numpy()[0]
      # Add results to the output.
      output.append({
          'preds': preds,
          'grads': grads,
          'grad_target': imagenet_labels.IMAGENET_2012_LABELS[grad_target_idx]
      })

    return output

  def input_spec(self):
    return {
        'image':
            lit_types.ImageBytes(),
        # If `grad_target` is not specified then the label with the highest
        # predicted score is used as the gradient target.
        'grad_target':
            lit_types.CategoryLabel(vocab=self.labels, required=False)
    }

  def output_spec(self):
    return {
        'preds':
            lit_types.MulticlassPreds(
                vocab=self.labels,
                autosort=True),
        'grads':
            lit_types.ImageGradients(
                align='image', grad_target_field_key='grad_target'),
        'grad_target':
            lit_types.CategoryLabel(vocab=self.labels)
    }

  def spec(self) -> model.ModelSpec:
    return self.MobileNetSpec(
        input=self.input_spec(), output=self.output_spec())
