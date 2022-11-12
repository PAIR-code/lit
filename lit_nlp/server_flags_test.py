"""Tests for server_flags."""

from absl import flags
from absl.testing import absltest
from lit_nlp import server_config
from lit_nlp import server_flags

FLAGS = flags.FLAGS


def _get_flags_for_module(module):
  """Get all of the flags defined in the specified module.

  This is very slow, but should be authoritative - so we use it in a test and
  rely on the SERVER_FLAGS list instead at runtime.

  Args:
    module: the module to get flags from

  Returns:
    dict mapping flag names (string) to values (various types).
  """
  ret = {}
  for name, value in FLAGS.flag_values_dict().items():
    if FLAGS.find_module_defining_flag(name) == module.__name__:
      ret[name] = value
  return ret


class ServerFlagsTest(absltest.TestCase):

  def setUp(self):
    super(ServerFlagsTest, self).setUp()
    FLAGS(['server_flags_test'])

  def test_matches_server_config(self):
    """Check that server_config and server_flags match."""
    server_config_dict = server_config.get_flags().to_dict()
    server_flags_dict = server_flags.get_flags()
    self.assertEqual(server_flags_dict, server_config_dict)

  def test_gets_all_flags(self):
    """Check that SERVER_FLAGS captures all the flags in server_flags.py."""
    all_server_flags = _get_flags_for_module(server_flags)
    server_flags_dict = server_flags.get_flags()
    self.assertEqual(server_flags_dict, all_server_flags)


if __name__ == '__main__':
  absltest.main()
