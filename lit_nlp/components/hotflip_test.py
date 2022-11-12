# Copyright 2020 Google LLC
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
"""Tests for lit_nlp.components.hotflip."""

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.components import hotflip
from lit_nlp.lib import utils
import numpy as np


class TestClassificationModel(lit_model.Model):

  def input_spec(self) -> dict[str, lit_types.LitType]:
    return {
        'sentence':
            lit_types.TextSegment(),
        'tokens_sentence':
            lit_types.Tokens(parent='sentence', required=False),
        'input_embs_sentence':
            lit_types.TokenEmbeddings(align='tokens_sentence', required=False),
    }

  def output_spec(self) -> dict[str, lit_types.LitType]:
    return {
        'probas':
            lit_types.MulticlassPreds(vocab=[]),
        'grad_class':
            lit_types.CategoryLabel(required=False, vocab=[]),
        'tokens':
            lit_types.Tokens(),
        'tokens_sentence':
            lit_types.Tokens(parent='sentence', required=False),
        'token_grad_sentence':
            lit_types.TokenGradients(
                align='tokens_sentence',
                grad_for='input_embs_sentence',
                grad_target_field_key='grad_class'),
    }

  def get_embedding_table(self):
    return ([], np.ndarray([]))

  def predict_minibatch(
      self, inputs: list[lit_model.JsonDict]) -> list[lit_model.JsonDict]:
    pass


class TestRegressionModel(lit_model.Model):

  def input_spec(self) -> dict[str, lit_types.LitType]:
    return {
        'sentence1':
            lit_types.TextSegment(),
        'tokens_sentence1':
            lit_types.Tokens(parent='sentence1', required=False),
        'input_embs_sentence1':
            lit_types.TokenEmbeddings(align='tokens_sentence1', required=False),
        'sentence2':
            lit_types.TextSegment(),
        'tokens_sentence2':
            lit_types.Tokens(parent='sentence2', required=False),
        'input_embs_sentence2':
            lit_types.TokenEmbeddings(align='tokens_sentence2', required=False),
    }

  def output_spec(self) -> dict[str, lit_types.LitType]:
    return {
        'score':
            lit_types.RegressionScore(),
        'grad_class':
            lit_types.CategoryLabel(required=False, vocab=[]),
        'tokens':
            lit_types.Tokens(),
        'tokens_sentence1':
            lit_types.Tokens(parent='sentence1', required=False),
        'token_grad_sentence1':
            lit_types.TokenGradients(
                align='tokens_sentence1',
                grad_for='input_embs_sentence1',
                grad_target_field_key='grad_class'),
        'tokens_sentence2':
            lit_types.Tokens(parent='sentence2', required=False),
        'token_grad_sentence2':
            lit_types.TokenGradients(
                align='tokens_sentence2',
                grad_for='input_embs_sentence2',
                grad_target_field_key='grad_class')
    }

  def get_embedding_table(self):
    return ([], np.ndarray([]))

  def predict_minibatch(
      self, inputs: list[lit_model.JsonDict]) -> list[lit_model.JsonDict]:
    pass


class ModelWithoutCallableEmbeddingTable():

  def __init__(self, base: lit_model.Model):
    self.get_embedding_table = 'get_embedding_table'
    self.base = base

  def input_spec(self):
    return self.base.input_spec()

  def output_spec(self):
    return self.base.output_spec()


class ModelWithoutEmbeddingTable():

  def __init__(self, base: lit_model.Model):
    self.base = base

  def input_spec(self):
    return self.base.input_spec()

  def output_spec(self):
    return self.base.output_spec()

  def get_embedding_table(self):
    raise NotImplementedError()


class ModelWithoutGradients():

  def __init__(self, base: lit_model.Model):
    self.base = base

  def input_spec(self):
    return self.base.input_spec()

  def output_spec(self):
    ret = self.base.output_spec()
    token_gradient_keys = utils.find_spec_keys(ret, lit_types.TokenGradients)
    for k in token_gradient_keys:
      ret.pop(k, None)
    return ret


class ModelWithoutTokens():

  def __init__(self, base: lit_model.Model):
    self.base = base

  def input_spec(self):
    ret = self.base.input_spec()
    token_keys = utils.find_spec_keys(ret, lit_types.Tokens)
    for k in token_keys:
      ret.pop(k, None)
    return ret

  def output_spec(self):
    return self.base.output_spec()


_CLASSIFICATION_GOOD = TestClassificationModel()
_CLASSIFICATION_NO_CALLABLE_EMB = ModelWithoutCallableEmbeddingTable(
    _CLASSIFICATION_GOOD)
_CLASSIFICATION_NO_EMBEDDING_TABLE = ModelWithoutEmbeddingTable(
    _CLASSIFICATION_GOOD)
_CLASSIFICATION_NO_GRADIENTS = ModelWithoutGradients(_CLASSIFICATION_GOOD)
_CLASSIFICATION_NO_TOKENS = ModelWithoutTokens(_CLASSIFICATION_GOOD)
_REGRESSION_GOOD = TestRegressionModel()
_REGRESSION_NO_CALLABLE_EMB = ModelWithoutCallableEmbeddingTable(
    _REGRESSION_GOOD)
_REGRESSION_NO_EMBEDDING_TABLE = ModelWithoutEmbeddingTable(_REGRESSION_GOOD)
_REGRESSION_NO_GRADIENTS = ModelWithoutGradients(_REGRESSION_GOOD)
_REGRESSION_NO_TOKENS = ModelWithoutTokens(_REGRESSION_GOOD)


class HotflipTest(parameterized.TestCase):

  def setUp(self):
    super(HotflipTest, self).setUp()
    self.hotflip = hotflip.HotFlip()

  @parameterized.named_parameters(
      ('cls', _CLASSIFICATION_GOOD, True),
      ('cls_no_call_emb_table', _CLASSIFICATION_NO_CALLABLE_EMB, False),
      ('cls_no_emb_table', _CLASSIFICATION_NO_EMBEDDING_TABLE, False),
      ('cls_no_gradients', _CLASSIFICATION_NO_GRADIENTS, False),
      ('cls_no_tokens', _CLASSIFICATION_NO_TOKENS, False),
      ('regression', _REGRESSION_GOOD, True),
      ('regression_no_call_emb_table', _REGRESSION_NO_CALLABLE_EMB, False),
      ('regression_no_emb_table', _REGRESSION_NO_EMBEDDING_TABLE, False),
      ('regression_no_gradients', _REGRESSION_NO_GRADIENTS, False),
      ('regression_no_tokens', _REGRESSION_NO_TOKENS, False),
  )
  def test_is_compatible(self, model: lit_model.Model, expected_compat: bool):
    compat = self.hotflip.is_compatible(
        model, lit_dataset.NoneDataset({'test': model}))
    self.assertEqual(expected_compat, compat)

  @parameterized.named_parameters(
      ('cls_no_align', _CLASSIFICATION_GOOD, lit_types.MulticlassPreds,
       ['probas'], None),
      ('cls_align', _CLASSIFICATION_GOOD, lit_types.TokenGradients,
       ['token_grad_sentence'], 'tokens_sentence'),
      ('cls_empty', _CLASSIFICATION_GOOD, lit_types.TokenGradients, [],
       'input_embs_sentence'),
      ('reg_no_align', _REGRESSION_GOOD, lit_types.RegressionScore, ['score'
                                                                    ], None),
      ('reg_align', _REGRESSION_GOOD, lit_types.TokenGradients,
       ['token_grad_sentence1'], 'tokens_sentence1'),
      ('reg_empty', _REGRESSION_GOOD, lit_types.TokenGradients, [],
       'input_embs_sentence1'),
  )
  def test_find_fields(self, model: lit_model.Model, to_find: lit_types.LitType,
                       expected_fields: list[str], align_field: Optional[str]):
    found = self.hotflip.find_fields(model.output_spec(), to_find, align_field)
    self.assertEqual(found, expected_fields)


if __name__ == '__main__':
  absltest.main()
