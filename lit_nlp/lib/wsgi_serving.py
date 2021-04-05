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

import socket
import threading
from typing import Optional, Text, List
from wsgiref import validate
import wsgiref.simple_server

from absl import logging
import portpicker
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
    """Start serving."""
    logging.info(('\n\nStarting Server on port %d'
                  '\nYou can navigate to %s:%d\n\n'), self._port, self._host,
                 self._port)
    werkzeug_serving.run_simple(
        self._host,
        self._port,
        self._app,
        use_debugger=False,
        use_reloader=False)


class WsgiServerIpv6(wsgiref.simple_server.WSGIServer):
  """IPv6 based extension of the simple WSGIServer."""

  address_family = socket.AF_INET6


class NotebookWsgiServer(object):
  """WSGI server for notebook environments."""

  def __init__(self, wsgi_app, host: Text = 'localhost',
               port: Optional[int] = None, **unused_kw):
    """Initialize the WSGI server.

    Args:
      wsgi_app: WSGI pep-333 application to run.
      host: Host to run on, defaults to 'localhost'.
      port: Port to run on. If not specified, then an unused one will be picked.
    """
    self._app = wsgi_app
    self._host = host
    self._port = port
    self._server_thread = None
    self.can_act_as_model_server = False

  @property
  def port(self):
    """Returns the current port or error if the server is not started.

    Raises:
      RuntimeError: If server has not been started yet.
    Returns:
      The port being used by the server.
    """
    if self._server_thread is None:
      raise RuntimeError('Server not started.')
    return self._port

  def stop(self):
    """Stops the server thread."""
    if self._server_thread is None:
      return
    self._stopping.set()
    self._server_thread = None
    self._stopped.wait()

  def serve(self):
    """Starts a server in a thread using the WSGI application provided.

    Will wait until the thread has started calling with an already serving
    application will simple return.
    """
    if self._server_thread is not None:
      return
    if self._port is None:
      self._port = portpicker.pick_unused_port()
    started = threading.Event()
    self._stopped = threading.Event()
    self._stopping = threading.Event()

    def build_server(started, stopped, stopping):
      """Closure to build the server function to be passed to the thread.

      Args:
        started: Threading event to notify when started.
        stopped: Threading event to notify when stopped.
        stopping: Threading event to notify when stopping.
      Returns:
        A function that function that takes a port and WSGI app and notifies
          about its status via the threading events provided.
      """

      def server(port, wsgi_app):
        """Serve a WSGI application until stopped.

        Args:
          port: Port number to serve on.
          wsgi_app: WSGI application to serve.
        """
        try:
          httpd = wsgiref.simple_server.make_server(self._host, port, wsgi_app)
        except socket.error:
          # Try IPv6
          httpd = wsgiref.simple_server.make_server(
              self._host, port, wsgi_app, server_class=WsgiServerIpv6)
        started.set()
        httpd.timeout = 30
        while not stopping.is_set():
          httpd.handle_request()
        stopped.set()

      return server

    server = build_server(started, self._stopped, self._stopping)
    server_thread = threading.Thread(
        target=server, args=(self._port, self._app))
    self._server_thread = server_thread

    server_thread.start()
    started.wait()
