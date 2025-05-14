import base64
from unittest import mock
from absl.testing import absltest
from vertexai import vision_models
from lit_nlp.examples.gcp_text_to_image import models


class MockModel:

  def __init__(
      self, images=None, raise_exception=False, sample_image_bytes=None
  ):
    self.images = images if images else []
    self.raise_exception = raise_exception
    self.call_count = 0
    self.sample_image_bytes = sample_image_bytes

  def generate_images(self, prompt, aspect_ratio=None):
    _, _ = prompt, aspect_ratio
    self.call_count += 1
    if self.raise_exception:
      raise ValueError("Mock Model Error")

    if self.sample_image_bytes:
      # Create a mock GeneratedImage instance, passing image_bytes
      mock_image = mock.create_autospec(
          vision_models.GeneratedImage, instance=True
      )
      mock_image._image_bytes = self.sample_image_bytes
      mock_response = vision_models.ImageGenerationResponse(images=[mock_image])
      return mock_response

    return vision_models.ImageGenerationResponse(images=[])


class ModelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Create a sample image for testing
    png_base64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAKh72VgAAAAASUVORK5CYII="
    self.sample_image_bytes = base64.b64decode(png_base64)

  @mock.patch(
      "vertexai.vision_models.ImageGenerationModel.from_pretrained",
  )
  @mock.patch("PIL.Image.open")
  def test_query_model(self, mock_image_open, mock_from_pretrained):
    # Create a MockModel instance
    mock_model = MockModel(
        sample_image_bytes=self.sample_image_bytes,
    )
    # Configure mock_from_pretrained to return the mock_model
    mock_from_pretrained.return_value = mock_model

    model = models.VertexModelGardenModel(model_name="test_model_name")
    mock_image = mock.Mock()

    mock_image.resize.return_value = mock_image
    mock_image_open.return_value = mock_image

    output = model.predict_minibatch(
        inputs=[{"prompt": "I say yes you say no"}]
    )
    result = list(output)

    self.assertLen(result, 1)
    self.assertIn("image", result[0])
    self.assertIn("prompt", result[0])
    self.assertEqual(result[0]["prompt"], "I say yes you say no")

    # Validate that the image is a base64 string
    self.assertTrue(result[0]["image"].startswith("data:image/png"))
    self.assertIsInstance(result[0]["image"], str)

    mock_from_pretrained.assert_called_once_with("test_model_name")

    # Assert that mock_generate_content was called
    self.assertEqual(mock_model.call_count, 1)


if __name__ == "__main__":
  absltest.main()
