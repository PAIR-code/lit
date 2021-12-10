"""Integration tests for lit_nlp.examples.models.glue_models."""

from absl.testing import absltest
from lit_nlp.examples.models import glue_models

import transformers


class GlueModelsIntTest(absltest.TestCase):

  def test_sst2_model_predict(self):
    # Create model.
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz"  # pylint: disable=line-too-long
    if model_path.endswith(".tar.gz"):
      model_path = transformers.file_utils.cached_path(
          model_path, extract_compressed_file=True)
    model = glue_models.SST2Model(model_path)

    # Run prediction to ensure no failure.
    model_in = [{"sentence": "test sentence"}]
    model_out = list(model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 1)
    for key in model.output_spec().keys():
      self.assertIn(key, model_out[0].keys())

if __name__ == "__main__":
  absltest.main()
