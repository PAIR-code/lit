# Lint as: python3
import collections
import functools
from absl.testing import absltest
from absl.testing import parameterized
from lime import lime_text as original_lime
from lit_nlp.components.citrus import lime
from lit_nlp.components.citrus import utils
import numpy as np
from scipy import special
from scipy import stats


class LimeTest(parameterized.TestCase):

  def test_sample_masks_returns_correct_shape_and_type(self):
    """Tests if LIME mask samples have the right shape and type."""
    num_samples = 2
    num_features = 3
    masks = lime.sample_masks(num_samples, num_features, seed=0)
    self.assertEqual(np.dtype('bool'), masks.dtype)
    self.assertEqual((num_samples, num_features), masks.shape)

  def test_sample_masks_contains_extreme_samples(self):
    """Tests if the masks contain extreme samples (1 or all features)."""
    num_samples = 1000
    num_features = 10
    masks = lime.sample_masks(num_samples, num_features, seed=0)
    num_disabled = (~masks).sum(axis=-1)
    self.assertEqual(1, min(num_disabled))
    self.assertEqual(num_features, max(num_disabled))

  def test_sample_masks_returns_uniformly_distributed_masks(self):
    """Tests if the masked positions are uniformly distributed."""
    num_samples = 10000
    num_features = 100
    masks = lime.sample_masks(num_samples, num_features, seed=0)
    # The mean should be ~0.5, but this is also true when normally distributed.
    np.testing.assert_almost_equal(masks.mean(), 0.5, decimal=2)
    # We should see each possible masked count approx. the same number of times.
    # We check this by looking at the entropy which should be around 1.0.
    counter = collections.Counter(masks.sum(axis=-1))
    entropy = stats.entropy(list(counter.values()), base=num_features)
    np.testing.assert_almost_equal(entropy, 1.0, decimal=2)

  def test_get_perturbations_returns_correctly_masked_string(self):
    """Tests obtaining perturbations from tokens and a mask."""
    sentence = 'It is a great movie but also somewhat bad .'
    tokens = sentence.split()
    # We create a mock mask with False for tokens with an 'a', True otherwise.
    masks = np.array([[False if 'a' in token else True for token in tokens]])
    perturbations = list(lime.get_perturbations(tokens, masks, mask_token='_'))
    expected = 'It is _ _ movie but _ _ _ .'
    self.assertEqual(expected, perturbations[0])

  @parameterized.named_parameters(
      {
          'testcase_name': 'is_one_for_zero_distance',
          'distance': 0.,
          'kernel_width': 10,
          'expected': 1.,
      }, {
          'testcase_name': 'is_zero_for_exp_kernel_width_distance',
          'distance': np.exp(10),
          'kernel_width': 10,
          'expected': 0.,
      })
  def test_exponential_kernel(self, distance, kernel_width, expected):
    """Tests a few known exponential kernel results."""
    result = lime.exponential_kernel(distance, kernel_width)
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'correctly_identifies_important_tokens_for_1d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'num_classes': 1,
          'class_to_explain': None,
      }, {
          'testcase_name': 'correctly_identifies_important_tokens_for_2d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'num_classes': 2,
          'class_to_explain': 1,
      }, {
          'testcase_name': 'correctly_identifies_important_tokens_for_3d_input',
          'sentence': 'It is a great movie but also somewhat bad .',
          'num_samples': 1000,
          'positive_token': 'great',
          'negative_token': 'bad',
          'num_classes': 3,
          'class_to_explain': 2,
      })
  def test_explain(self, sentence, num_samples, positive_token, negative_token,
                   num_classes, class_to_explain):
    """Tests explaining text classifiers with various output dimensions."""

    def _predict_fn(sentences):
      """Mock prediction function."""
      rs = np.random.RandomState(seed=0)
      predictions = []
      for sentence in sentences:
        probs = rs.uniform(0., 1., num_classes)
        # To check if LIME finds the right positive/negative correlations.
        if negative_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] - 1.
        if positive_token in sentence:
          probs[class_to_explain] = probs[class_to_explain] + 1.
        predictions.append(probs)

      predictions = np.stack(predictions, axis=0)
      if num_classes == 1:
        return np.squeeze(special.expit(predictions), -1)
      else:
        return special.softmax(predictions, axis=-1)

    explanation = lime.explain(
        sentence,
        _predict_fn,
        class_to_explain=class_to_explain,
        num_samples=num_samples,
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

  @parameterized.named_parameters({
      'testcase_name': 'correctly_identifies_important_tokens_for_regression',
      'sentence': 'It is a great movie but also somewhat bad .',
      'num_samples': 1000,
      'positive_token': 'great',
      'negative_token': 'bad',
  })
  def test_explain_regression(self, sentence, num_samples, positive_token,
                              negative_token):
    """Tests explaining text classifiers with various output dimensions."""

    def _predict_fn(sentences):
      """Mock prediction function."""
      rs = np.random.RandomState(seed=0)
      predictions = []
      for sentence in sentences:
        output = rs.uniform(-2., 2.)
        # To check if LIME finds the right positive/negative correlations.
        if negative_token in sentence:
          output -= rs.uniform(0., 2.)
        if positive_token in sentence:
          output += rs.uniform(0., 2.)
        predictions.append(output)

      predictions = np.stack(predictions, axis=0)
      return predictions

    explanation = lime.explain(
        sentence, _predict_fn, num_samples=num_samples, tokenizer=str.split)

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

    explanation = lime.explain('Test sentence', _predict_fn, 1, num_samples=5)
    self.assertNotEqual(explanation.intercept, 0.)

  def test_explain_returns_explanation_with_model(self):
    """Tests if the explanation contains the model."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lime.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_model=True)
    self.assertIsNotNone(explanation.model)

  def test_explain_returns_explanation_with_score(self):
    """Tests if the explanation contains a linear model score."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lime.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_score=True)
    self.assertIsNotNone(explanation.score)

  def test_explain_returns_explanation_with_prediction(self):
    """Tests if the explanation contains a prediction."""

    def _predict_fn(sentences):
      return np.random.uniform(0., 1., [len(list(sentences)), 2])

    explanation = lime.explain(
        'Test sentence',
        _predict_fn,
        class_to_explain=1,
        num_samples=5,
        return_prediction=True)
    self.assertIsNotNone(explanation.prediction)

  @parameterized.named_parameters(
      {
          'testcase_name': 'for_2d_input',
          'sentence': ' '.join(list('abcdefghijklmnopqrstuvwxyz')),
          'num_samples': 5000,
          'num_classes': 2,
          'class_to_explain': 1,
      }, {
          'testcase_name': 'for_3d_input',
          'sentence': ' '.join(list('abcdefghijklmnopqrstuvwxyz')),
          'num_samples': 5000,
          'num_classes': 3,
          'class_to_explain': 2,
      })
  def test_explain_matches_original_lime(self, sentence, num_samples,
                                         num_classes, class_to_explain):
    """Tests if Citrus LIME matches the original implementation."""
    list('abcdefghijklmnopqrstuvwxyz')
    # Assign some weight to each token a-z.
    # Each token contributes positively/negatively to the prediction.
    rs = np.random.RandomState(seed=0)
    token_weights = {token: rs.normal() for token in sentence.split()}
    token_weights[lime.DEFAULT_MASK_TOKEN] = 0.

    def _predict_fn(sentences):
      """Mock prediction function."""
      rs = np.random.RandomState(seed=0)
      predictions = []
      for sentence in sentences:
        probs = rs.normal(0., 0.1, size=num_classes)
        # To check if LIME finds the right positive/negative correlations.
        for token in sentence.split():
          probs[class_to_explain] += token_weights[token]
        predictions.append(probs)
      return np.stack(predictions, axis=0)

    # Explain the prediction using Citrus LIME.
    explanation = lime.explain(
        sentence,
        _predict_fn,
        class_to_explain=class_to_explain,
        num_samples=num_samples,
        tokenizer=str.split,
        mask_token=lime.DEFAULT_MASK_TOKEN,
        kernel=functools.partial(
            lime.exponential_kernel, kernel_width=lime.DEFAULT_KERNEL_WIDTH))
    scores = explanation.feature_importance  # <float32>[seq_len]
    scores = utils.normalize_scores(scores, make_positive=False)

    # Explain the prediction using original LIME.
    original_lime_explainer = original_lime.LimeTextExplainer(
        class_names=map(str, np.arange(num_classes)),
        mask_string=lime.DEFAULT_MASK_TOKEN,
        kernel_width=lime.DEFAULT_KERNEL_WIDTH,
        split_expression=str.split,
        bow=False)
    num_features = len(sentence.split())
    original_explanation = original_lime_explainer.explain_instance(
        sentence,
        _predict_fn,
        labels=(class_to_explain,),
        num_features=num_features,
        num_samples=num_samples)

    # original_explanation.local_exp is a dict that has a key class_to_explain,
    # which gives a sequence of (index, score) pairs.
    # We convert it to an array <float32>[seq_len] with a score per position.
    original_scores = np.zeros(num_features)
    for index, score in original_explanation.local_exp[class_to_explain]:
      original_scores[index] = score
    original_scores = utils.normalize_scores(
        original_scores, make_positive=False)

    # Test that Citrus LIME and original LIME match.
    np.testing.assert_allclose(scores, original_scores, atol=0.01)


if __name__ == '__main__':
  absltest.main()
