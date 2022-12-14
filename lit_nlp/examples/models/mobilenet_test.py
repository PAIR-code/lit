from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types
from lit_nlp.examples.models import mobilenet
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image as PILImage


class MobileNetTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = mobilenet.MobileNet()

  @parameterized.named_parameters(
      dict(
          testcase_name='compatible',
          dataset_spec={'image': types.ImageBytes()},
          expected=True,
      ),
      dict(
          testcase_name='empty',
          dataset_spec={},
          expected=False,
      ),
      dict(
          testcase_name='no_images',
          dataset_spec={'text': types.TextSegment()},
          expected=False,
      ),
      dict(
          testcase_name='wrong_keys',
          dataset_spec={'wrong_image_key': types.ImageBytes()},
          expected=False,
      ),
  )
  def test_compatibility(self, dataset_spec: types.Spec, expected: bool):
    dataset = lit_dataset.Dataset(spec=dataset_spec)
    self.assertEqual(self.model.is_compatible_with_dataset(dataset), expected)

  def test_model(self):
    # Create an input with base64 encoded image.
    input_1 = {
        'image': np.zeros(shape=(mobilenet.IMAGE_SHAPE), dtype=np.float32)
    }
    # Create an input with image data in Numpy array.
    pil_image = PILImage.new(mode='RGB', size=(300, 200))
    input_2 = {'image': image_utils.convert_pil_to_image_str(pil_image)}
    model_out = self.model.predict([input_1, input_2])
    model_out = list(model_out)
    # Check first output.
    self.assertIn('preds', model_out[0])
    self.assertIn('grads', model_out[0])
    self.assertIn('grad_target', model_out[0])
    # Check second output.
    self.assertIn('preds', model_out[1])
    self.assertIn('grads', model_out[1])
    self.assertIn('grad_target', model_out[1])


if __name__ == '__main__':
  absltest.main()
