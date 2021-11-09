# Lint as: python3
"""Local Interpretable Model-agnostic Explanations (LIME).

LIME was proposed in the following paper:

> "Why Should I Trust You?": Explaining the Predictions of Any Classifier
> Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
> https://arxiv.org/abs/1602.04938

LIME explains classifiers by returning a feature attribution score
for each input feature. It works as follows:

1) Sample perturbation masks. First the number of masked features is sampled
   (uniform, at least 1), and then that number of features are randomly chosen
   to be masked out (without replacement).
2) Get predictions from the model for those perturbations. Use these as labels.
3) Fit a linear model to associate the input positions indicated by the binary
   mask with the resulting predicted label.

The resulting feature importance scores are the linear model coefficients for
the requested output class or (in case of regression) the output score.

This is a reimplementation of the original https://github.com/marcotcr/lime
and is tested for compatibility. This version supports applying LIME to text
input, also in case of regression and binary-classification where the
prediction function only outputs a scalar for each input sentence.
"""
import functools
from typing import Any, Callable, Iterable, Optional, Sequence
from lit_nlp.components.citrus import helpers
import numpy as np
from sklearn import linear_model
from sklearn import metrics

DEFAULT_KERNEL_WIDTH = 25
DEFAULT_MASK_TOKEN = '<unk>'
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_SOLVER = 'cholesky'


def sample_masks(num_samples: int,
                 num_features: int,
                 seed: Optional[int] = None):
  """Samples LIME masks with at least 1 position disabled per sampled mask.

  The number of disabled features is sampled from a uniform distribution.

  Args:
    num_samples: The number of samples.
    num_features: The number of features to sample a mask for. Typically this is
      the number of tokens in the sentence.
    seed: Set this to an integer to make the sampling deterministic.

  Returns:
    Masks <bool>[num_samples, num_features] indicating which features are
    enabled (True) and which ones are disabled (False).
  """
  rng = np.random.RandomState(seed)
  positions = np.tile(np.arange(num_features), (num_samples, 1))
  permutation_fn = np.vectorize(rng.permutation, signature='(n)->(n)')
  permutations = permutation_fn(positions)  # A shuffled range of positions.
  num_disabled_features = rng.randint(1, num_features + 1, (num_samples, 1))
  # For num_disabled_features[i] == 2, this will set indices 0 and 1 to False.
  return permutations >= num_disabled_features


def get_perturbations(tokens: Sequence[str],
                      masks: np.ndarray,
                      mask_token: str = '<unk>') -> Iterable[str]:
  """Returns strings with the masked tokens replaced with `mask_token`."""
  for mask in masks:
    parts = [t if mask[i] else mask_token for i, t in enumerate(tokens)]
    yield ' '.join(parts)


def exponential_kernel(
    distance: float, kernel_width: float = DEFAULT_KERNEL_WIDTH) -> np.ndarray:
  """The exponential kernel."""
  return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def explain(
    sentence: str,
    predict_fn: Callable[[Iterable[str]], np.ndarray],
    class_to_explain: Optional[int] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    tokenizer: Any = str.split,
    mask_token: str = DEFAULT_MASK_TOKEN,
    alpha: float = 1.0,
    solver: str = DEFAULT_SOLVER,
    kernel: Callable[..., np.ndarray] = exponential_kernel,
    distance_fn: Callable[..., np.ndarray] = functools.partial(
        metrics.pairwise.pairwise_distances, metric='cosine'),
    distance_scale: float = 100.,
    return_model: bool = False,
    return_score: bool = False,
    return_prediction: bool = False,
    seed: Optional[int] = None,
) -> helpers.PosthocExplanation:
  """Returns the LIME explanation for a given sentence.

  By default, this function returns an explanation object containing feature
  importance scores and the intercept. Optionally, more information can be
  returned, such as the linear model, the score of the model on the perturbation
  set, and the prediction that the linear model makes on the original sentence.

  Args:
    sentence: An input to be explained.
    predict_fn: A prediction function that returns an array of outputs given a
      list of inputs. The output shape is [len(inputs)] for regression and
      binary classification (with scalar output), and [len(inputs), num_classes]
      for multi-class classification.
    class_to_explain: The class ID to explain in case of multi-class
      classification, where `predict_fn` returns outputs with multiple
      dimensions for each input. For example, use 2 to explain the third class
      in 3-class classification. For regression and binary classification, where
      `predict_fn` returns a scalar for each input, this does not need to be
      set.
    num_samples: The number of n-grams to sample.
    tokenizer: A function that splits the input sentence into tokens.
    mask_token: The token that is used for masking tokens, e.g., '<unk>'.
    alpha: Regularization strength of the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    solver: Solver to use in the linear approximation model. See
      `sklearn.linear_model.Ridge` for details.
    kernel: A kernel function to be used on the distance function. By default,
      the exponential kernel (with kernel width DEFAULT_KERNEL_WIDTH) is used.
    distance_fn: A distance function to use in range [0, 1]. Default: cosine.
    distance_scale: A scalar factor multiplied with the distances before the
      kernel is applied.
    return_model: Returns the fitted linear model.
    return_score: Returns the score of the linear model on the perturbations.
      This is the R^2 of the linear model predictions w.r.t. their targets.
    return_prediction: Returns the prediction of the linear model on the full
      original sentence.
    seed: Optional random seed to make the explanation deterministic.

  Returns:
    The explanation for the requested class.
  """
  # TODO(bastings): Provide sentence already tokenized to reduce split/join ops.
  tokens = tokenizer(sentence)

  masks = sample_masks(num_samples + 1, len(tokens), seed=seed)
  assert masks.shape[0] == num_samples + 1, 'Expected num_samples + 1 masks.'
  all_true_mask = np.ones_like(masks[0], dtype=np.bool)
  masks[0] = all_true_mask  # First mask is the full sentence.

  perturbations = list(get_perturbations(tokens, masks, mask_token))
  outputs = predict_fn(perturbations)

  if len(outputs.shape) > 1:
    assert class_to_explain is not None, \
        'class_to_explain needs to be set when `predict_fn` returns a 2D tensor'
    outputs = outputs[:, class_to_explain]  # We are only interested in 1 class.

  distances = distance_fn(all_true_mask.reshape(1, -1), masks).flatten()
  distances = distance_scale * distances
  distances = kernel(distances)

  # Fit a linear model for the requested output class.
  model = linear_model.Ridge(
      alpha=alpha, solver=solver, random_state=seed).fit(
          masks, outputs, sample_weight=distances)

  explanation = helpers.PosthocExplanation(
      feature_importance=model.coef_, intercept=model.intercept_)

  if return_model:
    explanation.model = model

  if return_score:
    explanation.score = model.score(masks, outputs)

  if return_prediction:
    explanation.prediction = model.predict(all_true_mask.reshape(1, -1))

  return explanation
