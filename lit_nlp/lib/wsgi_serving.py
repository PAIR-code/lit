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
"""WSGI servers to power the LIT backend."""

from typing import Optional, Text, List
from wsgiref import validate

from absl import logging
from werkzeug import serving as werkzeug_serving


class BasicDevServer(object):
  """Basic development server; not recommended for deployment."""

  def __init__(self, wsgi_app, port: int = 4321, host: Text = '127.0.0.1',
               **unused_kw):
    self._port = port
    self._host = host
    self._app = wsgi_app
    self.can_act_as_model_server = True

  def serve(self):
    logging.info(('\n\nStarting Server on port %d'
                  '\nYou can navigate to %s:%d\n\n'), self._port, self._host,
                 self._port)
    werkzeug_serving.run_simple(
        self._host,
        self._port,
        self._app,
        use_debugger=False,
        use_reloader=False)
