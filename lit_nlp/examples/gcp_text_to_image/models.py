"""Model Wrapper for generative models."""

from collections.abc import Iterable
import io
import logging
import time
from typing import Literal, Optional, Union
from vertexai import vision_models
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import image_utils
from PIL import Image

_MAX_NUM_RETRIES = 5

_DEFAULT_CANDIDATE_COUNT = 1

_DEFAULT_MAX_OUTPUT_TOKENS = 256

_IMAGE_PREFIX = 'data:image/png;base64,'


class VertexModelGardenModel(lit_model.BatchedRemoteModel):
  """VertexModelGardenModel is a wrapper for Vertex AI Model Garden model.

  Attributes:
    model_name: The name of the model to load.
    max_concurrent_requests: The maximum number of concurrent requests to the
      model.
    max_qps: The maximum number of queries per second to the model.
    temperature: The temperature to use for the model.
    candidate_count: The number of candidates to generate.
    max_output_tokens: The maximum number of tokens to generate.

  Please note the model will predict all examples at a fixed temperature.
  """

  def __init__(
      self,
      model_name: str = 'imagen-3.0-generate-002',
      max_concurrent_requests: int = 4,
      max_qps: Union[int, float] = 25,
      aspect_ratio: Optional[
          Literal['16:9', '1:1', '3:4', '4:3', '9:16']
      ] = None,
      width: int = 256,
      height: int = 256,
  ):
    super().__init__(max_concurrent_requests, max_qps)
    # Connect to the remote model.
    self._model = vision_models.ImageGenerationModel.from_pretrained(model_name)
    self._aspect_ratio = aspect_ratio
    self._width = width
    self._height = height

  def query_model(self, prompt: str, **unused_kw) -> list[lit_types.JsonDict]:
    num_attempts = 0
    predictions = None
    exception = None
    width = self._width
    height = self._height

    while num_attempts < _MAX_NUM_RETRIES and predictions is None:
      num_attempts += 1

      try:
        predictions = self._model.generate_images(
            prompt=prompt,
            aspect_ratio=self._aspect_ratio,
        )
      except Exception as e:  # pylint: disable=broad-except
        wait_time = 2**num_attempts
        exception = e
        logging.warning('Waiting %ds to retry... (%s)', wait_time, e)
        time.sleep(2**num_attempts)

    if predictions is None:
      raise ValueError(
          f'Failed to get predictions. ({exception})'
      ) from exception

    if not isinstance(predictions, Iterable):
      raise ValueError(f'Predictions is not an Iterable: {type(predictions)}')

    images = []
    for image_ in predictions.images:
      pil_img = Image.open(io.BytesIO(getattr(image_, '_image_bytes')))
      pil_img = pil_img.resize((width, height))
      images.append(image_utils.convert_pil_to_image_str(pil_img))

    return images

  def predict_minibatch(
      self, inputs: list[lit_types.JsonDict]
  ) -> list[lit_types.JsonDict]:
    """The model can generate up to 8 images per run, but LIT may only show one due to frontend limitations.

    In MinDalle demos, the grid_size parameter controls layout—for example,
    grid_size=2 creates a 2x2 grid of sub-images, rendered as a single final
    image. That’s why only one image might appear even if multiple are
    generated.

    Args:
      inputs: A list of input dictionaries, each containing a 'prompt'.

    Returns:
      A list of dictionaries, each containing the generated 'image' and the
      original 'prompt'.
    """
    results = []
    for inp in inputs:
      prompt = inp['prompt']
      b64_strs = self.query_model(prompt)
      if not b64_strs:
        raise ValueError(f'No images generated for prompt: {prompt}')
      results.append({
          'image': b64_strs[0],
          'prompt': prompt,
      })
    return results

  @classmethod
  def init_spec(cls) -> lit_types.Spec:
    return {
        'model_name': lit_types.String(
            default='imagen-3.0-generate-002', required=True
        ),
        'aspect_ratio': lit_types.String(default='1:1', required=False),
        'width': lit_types.Integer(default=256, required=False),
        'height': lit_types.Integer(default=256, required=False),
    }

  def input_spec(self) -> lit_types.Spec:
    return {
        'prompt': lit_types.TextSegment(),
    }

  def output_spec(self):
    return {
        'image': lit_types.ImageBytesList(),
    }
