"""Tests for lit_nlp.examples.models.glue_models."""

from absl.testing import absltest
from absl.testing import parameterized
import attr
from lit_nlp.examples.models import glue_models
import numpy as np


@attr.s(auto_attribs=True, kw_only=True)
class GlueModelConfigForTesting(object):
  num_hidden_layers: int = 3


@attr.s(auto_attribs=True, kw_only=True)
class GlueModelInternalForTesting(object):
  config = GlueModelConfigForTesting()


class GlueModelForTesting(glue_models.GlueModel):
  """Glue model for testing, which skips Huggingface initializations."""

  def _load_model(self, model_name_or_path):
    del model_name_or_path  # unused
    self.model = GlueModelInternalForTesting()


class GlueModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          config={},
          expect_attention=True,
          expect_embs=True,
          expect_grads=True,
      ),
      # Common individual cases
      dict(
          testcase_name="no_attention",
          config={"output_attention": False},
          expect_attention=False,
          expect_embs=True,
          expect_grads=True,
      ),
      dict(
          testcase_name="no_embeddings",
          config={"output_embeddings": False},
          expect_attention=True,
          expect_embs=False,
          expect_grads=True,
      ),
      dict(
          testcase_name="no_gradients",
          config={"compute_grads": False},
          expect_attention=True,
          expect_embs=True,
          expect_grads=False,
      ),
      # Common multiple cases
      dict(
          testcase_name="no_attention_or_embeddings",
          config={
              "output_attention": False,
              "output_embeddings": False
          },
          expect_attention=False,
          expect_embs=False,
          expect_grads=True,
      ),
      dict(
          testcase_name="no_attention_or_embeddings_or_gradients",
          config={
              "compute_grads": False,
              "output_attention": False,
              "output_embeddings": False
          },
          expect_attention=False,
          expect_embs=False,
          expect_grads=False,
      ),
  )
  def test_spec_affecting_config_options(self, config: dict[str, bool],
                                         expect_attention: bool,
                                         expect_embs: bool, expect_grads: bool):
    model = GlueModelForTesting(
        model_name_or_path="bert-base-uncased", **config)
    input_spec = model.input_spec()
    output_spec = model.output_spec()

    attention_fields = [key for key in output_spec if "/attention" in key]
    avg_emb_fields = [key for key in output_spec if "/avg_emb" in key]
    text_a_embs = "input_embs_" + model.config.text_a_name
    text_b_embs = "input_embs_" + model.config.text_b_name
    text_a_token_grads = "token_grad_" + model.config.text_a_name
    text_b_token_grads = "token_grad_" + model.config.text_b_name

    self.assertEqual(model.config.compute_grads, expect_grads)
    self.assertEqual(model.config.output_attention, expect_attention)
    self.assertEqual(model.config.output_embeddings, expect_embs)

    # Check required fields in input spec, should only be the text inputs.
    for key, field_spec in input_spec.items():
      if key == model.config.text_a_name or key == model.config.text_b_name:
        self.assertTrue(field_spec.required)
      else:
        self.assertFalse(field_spec.required)

    # Check required fields in output spec.
    for key, field_spec in output_spec.items():
      self.assertTrue(field_spec.required)

    if expect_attention:
      self.assertLen(attention_fields, model.model.config.num_hidden_layers)
    else:
      self.assertEmpty(attention_fields)

    if expect_embs:
      self.assertLen(avg_emb_fields, model.model.config.num_hidden_layers + 1)
      self.assertIn(text_a_embs, input_spec)
      self.assertIn(text_b_embs, input_spec)
      self.assertIn("cls_emb", output_spec)
      self.assertIn(text_a_embs, output_spec)
      self.assertIn(text_b_embs, output_spec)
    else:
      self.assertEmpty(avg_emb_fields)
      self.assertNotIn(text_a_embs, input_spec)
      self.assertNotIn(text_b_embs, input_spec)
      self.assertNotIn("cls_emb", output_spec)
      self.assertNotIn(text_a_embs, output_spec)
      self.assertNotIn(text_b_embs, output_spec)

    if expect_embs and expect_grads:
      self.assertIn(text_a_token_grads, output_spec)
      self.assertIn(text_b_token_grads, output_spec)
    else:
      self.assertNotIn(text_a_token_grads, output_spec)
      self.assertNotIn(text_b_token_grads, output_spec)

  def test_scatter_all_embeddings_single_input(self):
    glue_model = GlueModelForTesting(
        model_name_or_path="bert-base-uncased",
        text_a_name="sentence1")
    emb_size = 10
    # We'll inject zeros for the embeddings of 'hi',
    # while special tokens get vectors of 1s.
    embs_a = np.zeros((1, emb_size))
    input_embs = np.ones((1, 3, emb_size))
    # Scatter embs_a into input_embs
    result = glue_model.scatter_all_embeddings([{"sentence1": "hi",
                                                 "input_embs_sentence1": embs_a,
                                                 }], input_embs)
    target = [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]
    np.testing.assert_almost_equal(result, target)

  def test_scatter_all_embeddings_both_inputs(self):
    glue_model = GlueModelForTesting(
        model_name_or_path="bert-base-uncased",
        text_a_name="sentence1",
        text_b_name="sentence2")
    emb_size = 10
    # Inject zeros at positions corresponding to real tokens
    # in each segment. Special tokens get vectors of 1s.
    embs_a = np.zeros((1, emb_size))
    embs_b = np.zeros((3, emb_size))
    input_embs = np.ones((1, 7, emb_size))
    # Scatter embs_a and embs_b into input_embs
    result = glue_model.scatter_all_embeddings([{"sentence1": "hi",
                                                 "input_embs_sentence1": embs_a,
                                                 "sentence2": "how are you",
                                                 "input_embs_sentence2": embs_b
                                                 }], input_embs)
    target = [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]
    np.testing.assert_almost_equal(result, target)

  def test_scatter_all_embeddings_multi_batch(self):
    glue_model = GlueModelForTesting(
        model_name_or_path="bert-base-uncased",
        text_a_name="sentence1")
    emb_size = 4
    embs_a = np.zeros((1, emb_size))
    embs_b = np.zeros((2, emb_size))
    input_embs = np.ones((2, 4, emb_size))
    # Scatter embs_a and embs_b into input_embs
    result = glue_model.scatter_all_embeddings([{"sentence1": "hi",
                                                 "input_embs_sentence1": embs_a,
                                                 },
                                                {"sentence1": "hi there",
                                                 "input_embs_sentence1": embs_b,
                                                 }], input_embs)
    target = [[[1, 1, 1, 1],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [1, 1, 1, 1]],
              [[1, 1, 1, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1]]]
    np.testing.assert_almost_equal(result, target)

    # Scatter only embs_a into input_embs
    result = glue_model.scatter_all_embeddings([{"sentence1": "hi",
                                                 "input_embs_sentence1": embs_a,
                                                 },
                                                {"sentence1": "hi there"
                                                 }], input_embs)
    target = [[[1, 1, 1, 1],
               [0, 0, 0, 0],
               [1, 1, 1, 1],
               [1, 1, 1, 1]],
              [[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1]]]
    np.testing.assert_almost_equal(result, target)

    # Scatter only embs_b into input_embs
    result = glue_model.scatter_all_embeddings([{"sentence1": "hi"},
                                                {"sentence1": "hi there",
                                                 "input_embs_sentence1": embs_b,
                                                 }], input_embs)
    target = [[[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1]],
              [[1, 1, 1, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 1, 1]]]
    np.testing.assert_almost_equal(result, target)


if __name__ == "__main__":
  absltest.main()
