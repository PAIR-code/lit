"""Tests for lime_explainer."""

from absl.testing import absltest
from lit_nlp.components import lime_explainer

CLASS_KEY = lime_explainer.CLASS_KEY
KERNEL_WIDTH_KEY = lime_explainer.KERNEL_WIDTH_KEY
MASK_KEY = lime_explainer.MASK_KEY
NUM_SAMPLES_KEY = lime_explainer.NUM_SAMPLES_KEY
SEED_KEY = lime_explainer.SEED_KEY


class LimeExplainerTest(absltest.TestCase):

  def test_ig_init_args(self):
    test_config = {
        CLASS_KEY: '0',
        KERNEL_WIDTH_KEY: '128',
        MASK_KEY: '[UNK]',
        NUM_SAMPLES_KEY: '30',
        SEED_KEY: '0',
    }

    lime_default = lime_explainer.LIME()
    lime_autorun = lime_explainer.LIME(autorun=True)
    lime_config = lime_explainer.LIME(config=test_config)
    lime_all = lime_explainer.LIME(autorun=True, config=test_config)

    self.assertFalse(lime_default.meta_spec()['saliency'].autorun)
    self.assertEqual(lime_default.config_spec()[CLASS_KEY].default, '-1')
    self.assertEqual(lime_default.config_spec()[KERNEL_WIDTH_KEY].default,
                     '256')
    self.assertEqual(lime_default.config_spec()[MASK_KEY].default, '[MASK]')
    self.assertEqual(lime_default.config_spec()[NUM_SAMPLES_KEY].default, '256')
    self.assertEqual(lime_default.config_spec()[SEED_KEY].default, '')

    self.assertTrue(lime_autorun.meta_spec()['saliency'].autorun)
    self.assertEqual(lime_autorun.config_spec()[CLASS_KEY].default, '-1')
    self.assertEqual(lime_autorun.config_spec()[KERNEL_WIDTH_KEY].default,
                     '256')
    self.assertEqual(lime_autorun.config_spec()[MASK_KEY].default, '[MASK]')
    self.assertEqual(lime_autorun.config_spec()[NUM_SAMPLES_KEY].default, '256')
    self.assertEqual(lime_autorun.config_spec()[SEED_KEY].default, '')

    self.assertFalse(lime_config.meta_spec()['saliency'].autorun)
    self.assertEqual(lime_config.config_spec()[CLASS_KEY].default,
                     test_config[CLASS_KEY])
    self.assertEqual(lime_config.config_spec()[KERNEL_WIDTH_KEY].default,
                     test_config[KERNEL_WIDTH_KEY])
    self.assertEqual(lime_config.config_spec()[MASK_KEY].default,
                     test_config[MASK_KEY])
    self.assertEqual(lime_config.config_spec()[NUM_SAMPLES_KEY].default,
                     test_config[NUM_SAMPLES_KEY])
    self.assertEqual(lime_config.config_spec()[SEED_KEY].default,
                     test_config[SEED_KEY])

    self.assertTrue(lime_all.meta_spec()['saliency'].autorun)
    self.assertEqual(lime_all.config_spec()[CLASS_KEY].default,
                     test_config[CLASS_KEY])
    self.assertEqual(lime_all.config_spec()[KERNEL_WIDTH_KEY].default,
                     test_config[KERNEL_WIDTH_KEY])
    self.assertEqual(lime_all.config_spec()[MASK_KEY].default,
                     test_config[MASK_KEY])
    self.assertEqual(lime_all.config_spec()[NUM_SAMPLES_KEY].default,
                     test_config[NUM_SAMPLES_KEY])
    self.assertEqual(lime_all.config_spec()[SEED_KEY].default,
                     test_config[SEED_KEY])


if __name__ == '__main__':
  absltest.main()
