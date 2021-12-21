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
"""Development wrapper for LIT server."""
import importlib
import inspect

from absl import logging
from lit_nlp import app as lit_app
from lit_nlp.lib import wsgi_serving

WSGI_SERVERS = {}
WSGI_SERVERS['basic'] = wsgi_serving.BasicDevServer
WSGI_SERVERS['default'] = wsgi_serving.BasicDevServer
WSGI_SERVERS['notebook'] = wsgi_serving.NotebookWsgiServer


def get_lit_logo():
  """Prints the LIT logo as ASCII art."""
  # pyformat: disable
  logo = ('\n'
          r' (    (           ' '\n'
          r' )\ ) )\ )  *   ) ' '\n'
          r'(()/((()/(` )  /( ' '\n'
          r' /(_))/(_))( )(_))' '\n'
          r'(_)) (_)) (_(_()) ' '\n'
          r'| |  |_ _||_   _| ' '\n'
          r'| |__ | |   | |   ' '\n'
          r'|____|___|  |_|   ' '\n\n')
  # pyformat: enable
  return logo


def get_available_keywords(func):
  """Get names of keyword arguments to a function."""
  sig = inspect.signature(func)
  return [
      p.name
      for p in sig.parameters.values()
      if p.kind == p.POSITIONAL_OR_KEYWORD or p.kind == p.KEYWORD_ONLY
  ]


class Server(object):
  """Development version of LIT server.

  This wraps the real LIT server and allows for quick reloading of the server
  code without reloading models or datasets.
  """

  def __init__(self, *args, server_type='default', **kw):
    # We expose a single Server class to simplify client use, but internally
    # this is factored into a WSGI app (LitApp) and a webserver.
    # Positional arguments and some keywords passed to the LitApp,
    # which contains the LIT backend logic.
    self._app_args = args  # models, datasets, etc.
    self._app_kw = {
        k: kw.pop(k) for k in get_available_keywords(lit_app.LitApp) if k in kw
    }
    # Remaining keywords passed to the webserver class.
    self._server_kw = kw
    self._server_type = server_type

  def serve(self):
    """Run server, with optional reload loop and cache saving.

    If the server type is 'external', then the app is returned instead of
    served by this module.

    Returns:
      WSGI app if the server type is 'external', server if the server type
      is 'notebook', otherwise None when serving is complete.
    """

    while True:
      logging.info(get_lit_logo())
      logging.info('Starting LIT server...')
      app = lit_app.LitApp(*self._app_args, **self._app_kw)

      # If using a separate server program to serve the app, such as gunicorn,
      # then just return the WSGI app instead of serving it directly.
      if self._server_type == 'external':
        return app
      # Pre-bake mode runs any warm-start functions, saves the cache,
      # and exits. Designed to be used in container setup for faster launching.
      if self._server_type == 'prebake':
        app.save_cache()
        logging.info('Pre-bake completed; exiting server.')
        return

      server_fn = WSGI_SERVERS[self._server_type]
      server = server_fn(app, **self._server_kw)

      # server.serve isn't blocking for notebook server type.
      # For other types, the underlying server registers a SIGINT handler,
      # so if you hit Ctrl+C it will return.
      server.serve()
      if self._server_type == 'notebook':
        return server
      app.save_cache()
      # Optionally, reload server for development.
      # Potentially brittle - don't use this for real deployments.
      prompt = input('[Enter] to restart server, Q to quit.')
      if len(prompt) > 0:  # pylint: disable=g-explicit-length-test
        return

      ##
      # pylint: disable=g-import-not-at-top
      # Hot-reload Python sources for the next iteration.
      # Add any dependencies you want to reload here, as:
      #   from lit_nlp.components import foobar
      #   importlib.reload(foobar)
      from lit_nlp.components import projection
      importlib.reload(projection)
      from lit_nlp.components import pca
      importlib.reload(pca)
      # Be careful about reloading the 'types' module, since this will re-define
      # the type classes and can break code that relies on isinstance().

      # Finally, reload the lit_app library.
      importlib.reload(lit_app)
      # pylint: enable=g-import-not-at-top
