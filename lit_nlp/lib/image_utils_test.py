# Copyright 2021 Google LLC
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
"""Tests for lit_nlp.lib.image_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.lib import image_utils
import numpy as np
from PIL import Image as PILImage


def _create_image_array(shape: tuple[int, int, int]) -> np.ndarray[np.uint8]:
  height, width, channels = shape
  array = np.zeros(shape=shape, dtype=np.uint8)
  for i in range(1000):
    array[i % height, i % width, i % channels] = i % 256
  return array


_TEST_IMAGE_SHAPE = (100, 70, 3)
_TEST_IMAGE_ARRAY = _create_image_array(_TEST_IMAGE_SHAPE)
_TEST_IMAGE_RGB: PILImage.Image = PILImage.fromarray(_TEST_IMAGE_ARRAY)
_TEST_IMAGE_ARRAY_GREY: np.ndarray = np.asarray(_TEST_IMAGE_RGB.convert('L'))


class ImageUtilsTest(parameterized.TestCase):

  def test_pil_to_str(self):
    image_str = image_utils.convert_pil_to_image_str(_TEST_IMAGE_RGB)
    self.assertIsNotNone(image_str)

  def test_str_to_pil(self):
    image_str = image_utils.convert_pil_to_image_str(_TEST_IMAGE_RGB)
    pil_image = image_utils.convert_image_str_to_pil(image_str)
    image_array = np.asarray(pil_image)
    np.testing.assert_array_equal(_TEST_IMAGE_ARRAY, image_array)

  @parameterized.named_parameters(
      dict(
          testcase_name='greyscale_normalized',
          mode='L',
          normalize=True,
          expected=_TEST_IMAGE_ARRAY_GREY / 255,
      ),
      dict(
          testcase_name='greyscale_not_normalized',
          mode='L',
          normalize=False,
          expected=_TEST_IMAGE_ARRAY_GREY,
      ),
      dict(
          testcase_name='rgb_normalized',
          mode='RGB',
          normalize=True,
          expected=_TEST_IMAGE_ARRAY / 255,
      ),
      dict(
          testcase_name='rgb_not_normalized',
          mode='RGB',
          normalize=False,
          expected=_TEST_IMAGE_ARRAY,
      ),
  )
  def test_str_to_array(self, mode: str, normalize: bool, expected: np.ndarray):
    image_str = image_utils.convert_pil_to_image_str(_TEST_IMAGE_RGB)
    image_array = image_utils.convert_image_str_to_array(
        image_str, shape=_TEST_IMAGE_SHAPE, mode=mode, normalize=normalize
    )
    np.testing.assert_array_almost_equal(expected, image_array)

  def test_clip_unsigned_saliency(self):
    a = np.linspace(0, 100, num=101, endpoint=True)
    a_clipped = image_utils.clip_unsigned_saliency(a, fraction=0.1)
    self.assertEqual(a_clipped.min(), 0)
    self.assertEqual(a_clipped.max(), 90)
    self.assertLen(np.argwhere(a_clipped == 0), 1)
    self.assertLen(np.argwhere(a_clipped == 90), 11)

  def test_clip_signed_saliency(self):
    a = np.linspace(-50, 100, num=151, endpoint=True)
    a_clipped = image_utils.clip_signed_saliency(a, fraction=0.1)
    self.assertEqual(a_clipped.min(), -42)
    self.assertEqual(a_clipped.max(), 92)
    self.assertLen(np.argwhere(a_clipped == -42), 9)
    self.assertLen(np.argwhere(a_clipped == 92), 9)

  def test_normalize_unsigned_saliency(self):
    a = np.linspace(10, 100, num=101, endpoint=True)
    a_norm = image_utils.normalize_unsigned_saliency(a)
    self.assertAlmostEqual(a_norm.max(), 1.0)
    self.assertAlmostEqual(a_norm.min(), 0.0)

  def test_normalize_signed_saliency(self):
    # Test the case when the magnitude of positive numbers is higher.
    a = np.linspace(-10, 100, num=101, endpoint=True)
    a_norm = image_utils.normalize_signed_saliency(a)
    self.assertAlmostEqual(a_norm.max(), 1.0)
    self.assertAlmostEqual(a_norm.min(), 0.45)

    # Test the case when the magnitude of negative numbers is higher.
    a = np.linspace(-100, 10, num=101, endpoint=True)
    a_norm = image_utils.normalize_signed_saliency(a)
    self.assertAlmostEqual(a_norm.max(), 0.55)
    self.assertAlmostEqual(a_norm.min(), 0.0)

  def test_overlay_pixel_saliency(self):
    # Crate url encoded image representation.
    image_array = np.zeros(shape=(100, 70, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(image_array)
    image_str = image_utils.convert_pil_to_image_str(pil_image)

    # Create saliency.
    saliency = np.ones(shape=(50, 20), dtype=np.uint8)

    overlay_image = image_utils.overlay_pixel_saliency(
        image_str=image_str,
        saliency=saliency,
        cm_name='bwr',
        clip_fraction=0.01,
        alpha_mul=0.90,
        signed=True,
        pixel_saliency=True)
    self.assertIsNotNone(overlay_image)
    self.assertSequenceEqual((100, 70, 3), np.asarray(overlay_image).shape)

  def test_overlay_area_saliency(self):
    # Crate url encoded image representation.
    image_array = np.zeros(shape=(90, 70, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(image_array)
    image_str = image_utils.convert_pil_to_image_str(pil_image)

    # Create saliency.
    saliency = np.ones(shape=(50, 30), dtype=np.uint8)

    overlay_image = image_utils.overlay_pixel_saliency(
        image_str=image_str,
        saliency=saliency,
        cm_name='bwr',
        clip_fraction=0.01,
        alpha_mul=0.90,
        signed=False,
        pixel_saliency=False)
    self.assertIsNotNone(overlay_image)
    self.assertSequenceEqual((90, 70, 3), np.asarray(overlay_image).shape)


if __name__ == '__main__':
  absltest.main()
