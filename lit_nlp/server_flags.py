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
"""Common flags for the LIT server, for port, host, authentication, etc.

Not required to use LIT, but helpful as a convenience mixin.

Usage:
  server_kw = server_flags.get_flags()
  server = dev_server.Server(models, datasets, ..., **server_kw)
  server.serve()

TODO(lit-dev): consider defining a single ConfigDict instead of individual
flags.
"""

import os
import pathlib

from absl import flags

FLAGS = flags.FLAGS

##
# Server flags, passed to the WSGI server.
# LINT.IfChange
flags.DEFINE_integer('port', 5432, 'What port to serve on.')
flags.DEFINE_string(
    'server_type', 'default',
    'Webserver to use; see dev_server.py for options. Use "external" when '
    'using an external webserver like gunicorn, or "prebake" to run start-up '
    'tasks (like warm start and caching data) without starting a server.')
flags.DEFINE_string(
    'host', '127.0.0.1', 'What host address to serve on. Use 127.0.0.1 for '
    'local development, or 0.0.0.0 to allow external connections.')

##
# LIT application flags, passed to app.LitApp constructor.
flags.DEFINE_string(
    'data_dir', '', 'Directory to store/lookup persisted data used by server, '
    'such as cached predictions. If empty, will cache in-memory only.')
flags.DEFINE_float(
    'warm_start', 0.0,
    'If 1, will run all (model, dataset) on startup to populate the cache. '
    'If fractional, will only warm-start on a sample of each dataset, '
    'for development purposes.')
flags.DEFINE_bool(
    'warm_projections', False,
    'If true, will precompute server-side embedding projections such as PCA.')
flags.DEFINE_bool(
    'demo_mode', False,
    'If true, will disable capabilities not allowed in demo mode, such as '
    'saving generated datapoints to disk.')
flags.DEFINE_string(
    'default_layout', 'default',
    'Which layout to use by default (can be changed via url); see layout.ts')
flags.DEFINE_string(
    'canonical_url', None,
    'What url base to use when copying the LIT url (e.g., something other '
    'than just a local server address.')
flags.DEFINE_string('page_title', None, 'Custom page title for this server.')
flags.DEFINE_bool('development_demo', False, 'If true, signifies this LIT '
                  'instance is a development demo.')

flags.DEFINE_string(
    'client_root',
    os.path.join(
        pathlib.Path(__file__).parent.absolute(), 'client', 'build',
        'default'),
    'Path to frontend client.')
# LINT.ThenChange(server_config.py)


def get_flags():
  """Get all of the flags defined in this module.

  Returns:
    dict mapping flag names (string) to values (various types).
  """
  ret = {}
  for name, value in FLAGS.flag_values_dict().items():
    if FLAGS.find_module_defining_flag(name) == __name__:
      ret[name] = value
  return ret
