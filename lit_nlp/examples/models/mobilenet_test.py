from absl.testing import absltest
from lit_nlp.examples.models import mobilenet
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image as PILImage


class MyTestCase(absltest.TestCase):

  def test_model(self):
    model = mobilenet.MobileNet()
    # Create an input with base64 encoded image.
    input_1 = {
        'image': np.zeros(shape=(mobilenet.IMAGE_SHAPE), dtype=np.float32)
    }
    # Create an input with image data in Numpy array.
    pil_image = PILImage.new(mode='RGB', size=(300, 200))
    input_2 = {'image': image_utils.convert_pil_to_image_str(pil_image)}
    model_out = model.predict([input_1, input_2])
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
