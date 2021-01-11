"""Tests for lit_nlp.examples.models.glue_models."""

from absl.testing import absltest
from lit_nlp.examples.models import glue_models

import numpy as np


class GlueModelForTesting(glue_models.GlueModel):
  """Glue model for testing, which skips Huggingface initializations."""

  def _load_model(self, model_name_or_path):
    pass


class GlueModelsTest(absltest.TestCase):

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
