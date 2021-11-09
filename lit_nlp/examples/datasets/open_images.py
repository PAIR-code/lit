"""Open Images dataset from tfds."""

from lit_nlp.api import dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
from PIL import Image as PILImage
import tensorflow_datasets as tfds


class OpenImagesDataset(dataset.Dataset):
  """OpenImages TFDS dataset.

  See https://www.tensorflow.org/datasets/catalog/open_images_v4 for details.
  Images are at 72 JPEG quality and by default load 100 examples from the test
  set though this can be changed through the `split` parameter, as per TFDS
  documentation.
  """

  def __init__(self, split: str = 'test[:100]'):
    tfds_examples = tfds.as_numpy(
        tfds.load('open_images_v4/200k', split=split, download=True,
                  try_gcs=True))
    def convert_input(inp):
      pil_image = PILImage.fromarray(inp['image'])
      image_str = image_utils.convert_pil_to_image_str(pil_image)
      return {'image': image_str}
    self._examples = [convert_input(inp) for inp in tfds_examples]

  def spec(self) -> lit_types.Spec:
    return {
        'image': lit_types.ImageBytes(),
    }
