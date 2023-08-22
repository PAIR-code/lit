r"""Integration tests for lit_nlp.examples.models.glue_models.

Test locally with:

blaze test //third_party/py/lit_nlp/examples/models:integration_tests \
    --guitar_cluster=LOCAL \
    --test_output=streamed \
    --guitar_detach
"""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.examples.models import glue_models
from lit_nlp.lib import file_cache


# TODO(b/254110131): Fix test flakiness. Expand to SST-2, STS-B, and MNLI
class GlueModelsIntTest(parameterized.TestCase):

  def __init__(self, *args: Any, **kwargs: Any):
    super().__init__(*args, **kwargs)
    # Create the SST-2 Model
    model_path = "https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz"  # pylint: disable=line-too-long
    if model_path.endswith(".tar.gz"):
      model_path = file_cache.cached_path(
          model_path, extract_compressed_file=True)
    self.sst2_model = glue_models.SST2Model(model_path)

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          config={},
      ),
      # Common individual cases
      dict(
          testcase_name="no_attention",
          config={"output_attention": False},
      ),
      dict(
          testcase_name="no_embeddings",
          config={"output_embeddings": False},
      ),
      dict(
          testcase_name="no_gradients",
          config={"compute_grads": False},
      ),
      # Common multiple cases
      dict(
          testcase_name="no_attention_or_embeddings",
          config={
              "output_attention": False,
              "output_embeddings": False
          },
      ),
      dict(
          testcase_name="no_attention_or_embeddings_or_gradients",
          config={
              "compute_grads": False,
              "output_attention": False,
              "output_embeddings": False
          },
      ),
  )
  def test_sst2_model_predict(self, config: dict[str, bool]):
    # Configure model.
    if config:
      self.sst2_model.config = glue_models.GlueModelConfig(
          # Include the SST-2 defaut config options
          text_a_name="sentence",
          text_b_name=None,
          labels=["0", "1"],
          null_label_idx=0,
          # Add the output-affecting config options
          **config)

    # Run prediction to ensure no failure.
    model_in = [{"sentence": "test sentence"}]
    model_out = list(self.sst2_model.predict(model_in))

    # Sanity-check output vs output spec.
    self.assertLen(model_out, 1)
    for key in self.sst2_model.output_spec().keys():
      self.assertIn(key, model_out[0])

if __name__ == "__main__":
  absltest.main()
