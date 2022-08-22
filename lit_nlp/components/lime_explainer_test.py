"""Tests for lime_explainer."""

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.components import lime_explainer

CLASS_KEY = lime_explainer.CLASS_KEY
KERNEL_WIDTH_KEY = lime_explainer.KERNEL_WIDTH_KEY
MASK_KEY = lime_explainer.MASK_KEY
NUM_SAMPLES_KEY = lime_explainer.NUM_SAMPLES_KEY
SEED_KEY = lime_explainer.SEED_KEY


class LimeExplainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', None, None, None, None, None, None),
      ('autorun only', True, None, None, None, None, None),
      ('config values only', None, 0, 128, '<UNK>', 128, 0),
      ('autorun + config values', True, 0, 128, '<UNK>', 128, 0))
  def test_lime_init_args(self, autorun, class_index, kernel_width, mask_token,
                          num_samples, seed):
    params = {}
    if autorun is not None: params['autorun'] = autorun
    if class_index is not None: params['class_index'] = class_index
    if kernel_width is not None: params['kernel_width'] = kernel_width
    if mask_token is not None: params['mask_token'] = mask_token
    if num_samples is not None: params['num_samples'] = num_samples
    if seed is not None: params['seed'] = seed

    lime = lime_explainer.LIME(**params)
    config_spec = lime.config_spec()

    if autorun is not None:
      self.assertEqual(lime.meta_spec()['saliency'].autorun, autorun)
    if class_index is not None:
      self.assertEqual(config_spec[CLASS_KEY].default, str(class_index))
    if kernel_width is not None:
      self.assertEqual(config_spec[KERNEL_WIDTH_KEY].default, str(kernel_width))
    if mask_token is not None:
      self.assertEqual(config_spec[MASK_KEY].default, str(mask_token))
    if num_samples is not None:
      self.assertEqual(config_spec[NUM_SAMPLES_KEY].default, str(num_samples))
    if seed is not None:
      self.assertEqual(config_spec[SEED_KEY].default, str(seed))


if __name__ == '__main__':
  absltest.main()
