from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types
from lit_nlp.examples.models import mobilenet
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image as PILImage


class MobileNetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='compatible_spec_model_v2',
          model_name='mobilenet_v2',
          dataset_spec={'image': types.ImageBytes()},
          expected=True,
      ),
      dict(
          testcase_name='empty_spec_model_v2',
          model_name='mobilenet_v2',
          dataset_spec={},
          expected=False,
      ),
      dict(
          testcase_name='no_images_spec_model_v2',
          model_name='mobilenet_v2',
          dataset_spec={'text': types.TextSegment()},
          expected=False,
      ),
      dict(
          testcase_name='wrong_keys_spec_model_v2',
          model_name='mobilenet_v2',
          dataset_spec={'wrong_image_key': types.ImageBytes()},
          expected=False,
      ),
      dict(
          testcase_name='compatible_spec_model_v1',
          model_name='mobilenet',
          dataset_spec={'image': types.ImageBytes()},
          expected=True,
      ),
      dict(
          testcase_name='empty_spec_model_v1',
          model_name='mobilenet',
          dataset_spec={},
          expected=False,
      ),
      dict(
          testcase_name='no_images_spec_model_v1',
          model_name='mobilenet',
          dataset_spec={'text': types.TextSegment()},
          expected=False,
      ),
      dict(
          testcase_name='wrong_keys_spec_model_v1',
          model_name='mobilenet',
          dataset_spec={'wrong_image_key': types.ImageBytes()},
          expected=False,
      ),
  )
  def test_compatibility(
      self, model_name: str, dataset_spec: types.Spec, expected: bool
  ):
    dataset = lit_dataset.Dataset(spec=dataset_spec)
    model = mobilenet.MobileNet(model_name)
    self.assertEqual(model.is_compatible_with_dataset(dataset), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='model_v1',
          model_name='mobilenet',
      ),
      dict(
          testcase_name='model_v2',
          model_name='mobilenet_v2',
      ),
  )
  def test_model(self, model_name: str):
    # Create an input with base64 encoded image.
    input_1 = {
        'image': np.zeros(shape=(mobilenet.IMAGE_SHAPE), dtype=np.float32)
    }
    # Create an input with image data in Numpy array.
    pil_image = PILImage.new(mode='RGB', size=(300, 200))
    input_2 = {'image': image_utils.convert_pil_to_image_str(pil_image)}
    model = mobilenet.MobileNet(model_name)
    model_out = model.predict([input_1, input_2])
    model_out = list(model_out)
    # Check first output.
    self.assertIn('preds', model_out[0])
    self.assertIn('grads', model_out[0])
    # Check second output.
    self.assertIn('preds', model_out[1])
    self.assertIn('grads', model_out[1])


if __name__ == '__main__':
  absltest.main()
