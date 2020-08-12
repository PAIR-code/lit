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
# Lint as: python3
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.components.citrus import lemon
import numpy as np
from scipy import special


class LemonTest(parameterized.TestCase):

  def test_make_vocab_dict(self):
    """Tests if the output of make_vocab_dict has the correct length and values.
    """
    tokens = ['abc', 'def', 'abc']
    vocab_dict = lemon.make_vocab_dict(tokens)

    self.assertLen(vocab_dict, 2)
    self.assertEqual(vocab_dict[tokens[0]], [0, 2])
    self.assertEqual(vocab_dict[tokens[1]], [1])

  def test_get_masks(self):
    """Tests if masks have the correct length and values."""
    tokens = ['a', 'b', 'c', 'c']
    vocab_dict = lemon.make_vocab_dict(tokens)

    counterfactuals = [['a'], ['b', 'c'], ['b', 'd', 'a']]
    masks = lemon.get_masks(counterfactuals, vocab_dict)

    self.assertLen(masks, 3)
    self.assertEqual(masks[0], [True, False, False])
    self.assertEqual(masks[1], [False, True, True])
    self.assertEqual(masks[2], [True, True, False])

  @parameterized.named_parameters(
      {
          'testcase_name': 'correctly_identifies_important_tokens_for_1d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'counterfactuals': ['It is an ok movie but also somewhat bad .',
                              'It is a terrible movie but also somewhat bad .',
                              'It is a good movie but also somewhat bad .',
                              'It was a good movie but also somewhat bad .',
                              'It was a great film but also somewhat bad .',
                              'It was a great show but also somewhat bad .',
                              'It was the great movie but also somewhat bad .',
                              'It was a movie but somewhat bad .',
                              'It was a movie and also somewhat bad .',
                              'It was a movie but also very bad .',
                              'It was a great but also bad .',
                              'There is a good movie but also somewhat bad .',
                              'is a great movie but also somewhat bad .',
                              'is a great movie but also somewhat .',
                              'is a great movie also somewhat bad .',
                              'is a great also somewhat .'],
          'positive_token': 'great',
          'negative_token': 'bad',
          'num_classes': 1,
          'class_to_explain': 0,
      }
      ,
      {
          'testcase_name':
              'correctly_identifies_important_tokens_for_input_with_duplicates',
          'sentence': 'It is a great movie but it is also somewhat bad .',
          'counterfactuals':
              ['It is an ok movie but its also somewhat bad .',
               'It is a terrible movie but it is also somewhat bad .',
               'It is a good movie but it is also somewhat bad .',
               'It was a good movie but it is also somewhat bad .',
               'It was a great film but it is also somewhat bad .',
               'It was a great show but it is bad also somewhat bad .',
               'It was the great movie but it is also somewhat bad .',
               'It was a movie but is somewhat bad .',
               'It was a movie and also it is somewhat bad .',
               'It was a movie but also it is very bad .',
               'It was a great but also it is bad .',
               'There is a good movie but also is somewhat bad .',
               'is a great movie but also it is somewhat bad .',
               'is a great movie but also it is somewhat .',
               'is a great movie also it is somewhat bad .',
               'is a great also it is somewhat .'],
          'positive_token': 'great',
          'negative_token': 'bad',
          'num_classes': 1,
          'class_to_explain': 0,
      }
      )
  def test_explain(self, sentence, counterfactuals, positive_token,
                   negative_token, num_classes, class_to_explain):
    """Tests explaining text classifiers with various output dimensions."""

    def _predict_fn(sentences):
      """Mock prediction function."""
      predictions = []
      np.random.seed(0)
      for sentence in sentences:
        probs = np.random.uniform(0., 1., num_classes)
        # To check if LEMON finds the right positive/negative correlations.
        if negative_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] - 1.
        if positive_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] + 1.
        predictions.append(probs)

      predictions = np.stack(predictions, axis=0)
      if num_classes == 1:
        return special.expit(predictions)
      else:
        return special.softmax(predictions, axis=-1)

    explanation = lemon.explain(
        sentence,
        counterfactuals,
        _predict_fn,
        class_to_explain,
        tokenizer=str.split)

    self.assertLen(explanation.feature_importance, len(sentence.split()))

    # The positive word should have the highest attribution score.
    positive_token_idx = sentence.split().index(positive_token)
    self.assertEqual(positive_token_idx,
                     np.argmax(explanation.feature_importance))

    # The negative word should have the lowest attribution score.
    negative_token_idx = sentence.split().index(negative_token)
    self.assertEqual(negative_token_idx,
                     np.argmin(explanation.feature_importance))

  def test_explain_returns_explanation_with_intercept(self):
    """Tests if the explanation contains an intercept value."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lemon.explain(
        'Test sentence',
        ['Test counterfactual'],
        _predict_fn,
        class_to_explain=1)
    self.assertNotEqual(explanation.intercept, 0.)

  def test_explain_returns_explanation_with_model(self):
    """Tests if the explanation contains the model."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lemon.explain(
        'Test sentence',
        ['Test counterfactual'],
        _predict_fn,
        class_to_explain=1,
        return_model=True)
    self.assertIsNotNone(explanation.model)

  def test_explain_returns_explanation_with_score(self):
    """Tests if the explanation contains a linear model score."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lemon.explain(
        'Test sentence',
        ['Test counterfactual'],
        _predict_fn,
        class_to_explain=1,
        return_score=True)
    self.assertIsNotNone(explanation.score)

  def test_explain_returns_explanation_with_prediction(self):
    """Tests if the explanation contains a prediction."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lemon.explain(
        'Test sentence',
        ['Test counterfactual'],
        _predict_fn,
        class_to_explain=1,
        return_prediction=True)
    self.assertIsNotNone(explanation.prediction)

  def test_duplicate_tokens(self):
    """Checks the explanation for a sentence with duplicate tokens."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    sentence = 'it is a great movie but it is also somewhat bad .'
    counterfactuals = ['it is an ok movie but its also somewhat bad .',
                       'it is a terrible movie but it is also somewhat bad .',
                       'it is a good movie but it is also somewhat bad .',
                       'it was a good movie but it also somewhat bad .',
                       'it was a great film but it is also somewhat bad .',
                       'it was a great show but it is bad also somewhat bad .',
                       'it was the great movie but it is also somewhat bad .',
                       'it was a movie but is somewhat bad .',
                       'it was a movie and also it is somewhat bad .',
                       'it was a movie but also it is very bad .',
                       'it was a great but also it is bad .',
                       'There is a good movie but also is somewhat bad .',
                       'is a great movie but also it somewhat bad .',
                       'is a great movie but also it is somewhat .',
                       'is a great movie also it is somewhat bad .',
                       'is a great also it is somewhat .']
    explanation = lemon.explain(
        sentence,
        counterfactuals,
        _predict_fn,
        class_to_explain=1,
        return_model=True)

    # Check that the number of model coefficients is equal to the number of
    # unique tokens in the original sentence.
    tokens = sentence.split()
    unique_tokens = set(tokens)
    self.assertLen(explanation.model.coef_, len(unique_tokens))

    # Check that the importance value for 'it' and 'it' are the same.
    self.assertEqual(explanation.feature_importance[0],
                     explanation.feature_importance[6])

    # Check that the importance value for 'is' and 'is' are the same.
    self.assertEqual(explanation.feature_importance[1],
                     explanation.feature_importance[7])

    print(explanation.feature_importance)

  def test_lowercase_tokens(self):
    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    sentence = 'It is a great movie but it is also somewhat bad .'
    counterfactuals = ['It is an ok movie but its also somewhat bad .',
                       'It is a terrible movie but it is also somewhat bad .',
                       'It is a good movie but it is also somewhat bad .',
                       'It was a good movie but it is also somewhat bad .',
                       'It was a great film but it is also somewhat bad .',
                       'It was a great show but it is bad also somewhat bad .',
                       'It was the great movie but it is also somewhat bad .',
                       'It was a movie but is somewhat bad .',
                       'It was a movie and also it is somewhat bad .',
                       'It was a movie but also it is very bad .',
                       'It was a great but also it is bad .',
                       'There is a good movie but also is somewhat bad .',
                       'is a great movie but also it is somewhat bad .',
                       'is a great movie but also it is somewhat .',
                       'is a great movie also it is somewhat bad .',
                       'is a great also it is somewhat .']

    explanation_lowercase = lemon.explain(
        sentence,
        counterfactuals,
        _predict_fn,
        class_to_explain=1,
        lowercase_tokens=True,
        return_model=True)

    # Check that the number of model coefficients is equal to the number of
    # unique tokens in the original sentence.
    tokens = [token.lower() for token in sentence.split()]
    unique_tokens = set(tokens)
    self.assertLen(explanation_lowercase.model.coef_, len(unique_tokens))

    # Check that the importance value for 'It' and 'it' are the same.
    self.assertEqual(explanation_lowercase.feature_importance[0],
                     explanation_lowercase.feature_importance[6])

    explanation_not_lowercase = lemon.explain(
        sentence,
        counterfactuals,
        _predict_fn,
        class_to_explain=1,
        lowercase_tokens=False,
        return_model=True)

    # Check that the number of model coefficients is equal to the number of
    # unique tokens in the original sentence.
    tokens = sentence.split()
    unique_tokens = set(tokens)
    self.assertLen(explanation_not_lowercase.model.coef_, len(unique_tokens))

    # Check that the importance value for 'It' and 'it' are not the same.
    self.assertNotEqual(explanation_not_lowercase.feature_importance[0],
                        explanation_not_lowercase.feature_importance[6])


if __name__ == '__main__':
  absltest.main()
