"""Contains utility methods used by the image demo app."""
import base64
import io
from typing import Tuple

import matplotlib.cm as plt_cm
import numpy as np
from PIL import Image as PILImage
from PIL import ImageEnhance as PILImageEnhance


def convert_image_str_to_pil(image_str: str) -> PILImage.Image:
  # Convert base64 string to PIL image.
  image_str = image_str[image_str.find(';base64,') + 8:]
  img_bytes = base64.b64decode(image_str.encode())
  return PILImage.open(io.BytesIO(img_bytes))


def convert_image_str_to_array(image_str: str, shape: Tuple[int, int,
                                                            int]) -> np.ndarray:
  """Converts a base64 encoded image to numpy array."""
  pil_image = convert_image_str_to_pil(image_str)
  # Resize image to match the model internal image size.
  pil_image = pil_image.resize((shape[1], shape[0]), PILImage.BILINEAR)
  # Convert image to the model format.
  pil_image = pil_image.convert(mode='RGB')
  # Return image data as an array.
  return np.asarray(pil_image, dtype=np.float32) / 255


def convert_pil_to_image_str(pil_image: PILImage.Image) -> str:
  """Converts PIL image to base64 URL encoded string."""
  buffered = io.BytesIO()
  pil_image.save(buffered, format='PNG')
  img_str = base64.b64encode(buffered.getvalue())
  return 'data:image/png;base64,' + img_str.decode('utf-8')


def normalize_signed_saliency(saliency: np.ndarray) -> np.ndarray:
  """Normalizes saliency map while preserving the sign and relative ratios.

  All result values are in interval [0, 1] but may assume a narrower interval.
  Value 0 in the original saliency array is mapped to 0.5 in the result
  saliency. E.g. the normalization of values in range [-10, 100] results in the
  output, which values belong to interval [0,45, 1.0].

  Args:
    saliency: a saliency map that should be normalized.

  Returns:
    The normalized saliency map.
  """
  saliency = saliency.astype(np.float32)
  max_abs_val = np.abs(saliency).max()
  if max_abs_val > 0:
    return (saliency / max_abs_val) / 2 + 0.5
  else:
    return saliency


def normalize_unsigned_saliency(saliency: np.ndarray) -> np.ndarray:
  """Normalizes positive only saliency map to range [0, 1]."""
  assert saliency.min() >= 0
  saliency = saliency.astype(np.float32)
  saliency = saliency - saliency.min()
  max_val = saliency.max()
  if max_val > 0:
    return saliency / max_val
  else:
    return saliency


def clip_signed_saliency(saliency: np.ndarray, fraction=0.01) -> np.ndarray:
  """Clips top and bottom parts if a signed saliency map."""
  b_value = np.quantile(saliency, fraction / 2, interpolation='higher')
  t_value = np.quantile(saliency, 1 - fraction / 2, interpolation='lower')
  return np.clip(saliency, min(0, b_value), max(0, t_value))


def clip_unsigned_saliency(saliency: np.ndarray, fraction=0.01) -> np.ndarray:
  """Clips the top part if an unsigned saliency map."""
  assert saliency.min() >= 0
  t_value = np.quantile(saliency, 1 - fraction, interpolation='lower')
  return np.clip(saliency, 0, t_value)


def overlay_pixel_saliency(image_str: str, saliency: np.ndarray, cm_name: str,
                           clip_fraction: float, alpha_mul: float, signed: bool,
                           pixel_saliency: bool) -> PILImage.Image:
  """Overlays saliency data on top of the input image."""

  # Convert original image to PIL.
  img = convert_image_str_to_pil(image_str)
  img = img.convert(mode='RGBA')

  # Normalize saliency values.
  if signed:
    clipped_saliency = clip_signed_saliency(saliency, fraction=clip_fraction)
    norm_saliency = normalize_signed_saliency(clipped_saliency)
  else:
    saliency = np.abs(saliency)
    clipped_saliency = clip_unsigned_saliency(saliency, fraction=clip_fraction)
    norm_saliency = normalize_unsigned_saliency(clipped_saliency)

  # Map saliency to RGB values.
  cm = plt_cm.get_cmap(cm_name)
  saliency_bytes = cm(norm_saliency)

  # Assign alpha values.
  if pixel_saliency:
    if signed:
      alphas = map(lambda e: abs(e - 0.5) * 2, norm_saliency.flatten())
    else:
      alphas = map(lambda e: 1.0 - e, norm_saliency.flatten())
    alphas = np.reshape(list(alphas), newshape=norm_saliency.shape)
  else:
    alphas = 1.0

  alphas *= alpha_mul
  saliency_bytes[:, :, 3] = alphas

  # Adjust the original image brightness.
  brightness_enhancer = PILImageEnhance.Brightness(img)
  img = brightness_enhancer.enhance(0.5)
  color_enhancer = PILImageEnhance.Color(img)
  img = color_enhancer.enhance(0.0)

  # Overlay original image with the saliency heatmap.
  saliency_bytes = (saliency_bytes * 255).astype(np.uint8)
  saliency_img = PILImage.fromarray(saliency_bytes, mode='RGBA')
  saliency_img = saliency_img.resize(size=img.size, resample=PILImage.BILINEAR)
  heatmap_img = PILImage.alpha_composite(img, saliency_img)
  heatmap_img = heatmap_img.convert(mode='RGB')
  return heatmap_img
