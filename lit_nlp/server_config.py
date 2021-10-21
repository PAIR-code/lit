"""Common flags for the LIT server, for port, host, authentication, etc.

Not required to use LIT, but helpful as a convenience mixin.

Usage:
  server_kw = config_flags.get_flags()
  server = dev_server.Server(models, datasets, ..., **server_kw)
  server.serve()

  On the commandline: --lit.port=5432 (instead of --port=5432)

A fork of google3/third_party/py/lit_nlp/server_flags.py, which it will
eventually replace: since absl.FLAGS is global, so importing server_flags.py
directly causes conflicts when LIT is run in the same binary as something else
that has the same flag keys.

TODO: migrate demos over to use this instead of server_flags.py.
"""

import ml_collections
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

config = ml_collections.ConfigDict()

##
# Server flags, passed to the WSGI server.
# LINT.IfChange

# What port to serve on.
config.port = 5432

# Webserver to use; see dev_server.py for options. Use "external" when
# using an external webserver like gunicorn, or "prebake" to run start-up
# tasks (like warm start and caching data) without starting a server.
config.server_type = 'default'

# What host address to serve on. Use 127.0.0.1 for local development, or
# 0.0.0.0 to allow external connections.'
config.host = '127.0.0.1'

##
# LIT application flags, passed to app.LitApp constructor.
# Directory to store/lookup persisted data used by server,
# such as cached predictions. If empty, will cache in-memory only.
config.data_dir = ''

# If 1, will run all (model, dataset) on startup to populate the cache.
# If fractional, will only warm-start on a sample of each dataset,
# for development purposes.
config.warm_start = 0.0

# If true, will precompute server-side embedding projections such as PCA.
config.warm_projections = config_dict.placeholder(bool)

# If true, will disable capabilities not allowed in demo mode, such as
# saving generated datapoints to disk.
config.demo_mode = config_dict.placeholder(bool)

# Which layout to use by default (can be changed via url); see layout.ts
config.default_layout = 'default'

# What url base to use when copying the LIT url (e.g., something other
# than just a local server address.
config.canonical_url = config_dict.placeholder(str)

# Custom page title for this server.
config.page_title = config_dict.placeholder(str)

# Whether the LIT instance is a development demo.
config.development_demo = False

config.client_root = os.path.join(
        pathlib.Path(__file__).parent.absolute(), 'client', 'build',
        'default')

config_flags.DEFINE_config_dict('lit', config)
# LINT.ThenChange(server_flags.py)


def get_flags():
  return config
