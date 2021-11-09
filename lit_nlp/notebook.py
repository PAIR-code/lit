"""Notebook usage of LIT.

To use in LIT in colab or jupyter notebooks, create a LitWidget instance
with models and datasets to load. Optionally set the UI height and a proxy URL
if necessary. By default, the UI with render in the cell that creates the
instance. Set render=False to disable this, and manually render the UI in a cell
through the render() method. Use the stop() method to stop the server when done.
"""

import html
import json
import os
import pathlib
import random
import typing
# pytype: disable=import-error
from IPython import display
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.lib import wsgi_serving

try:
  import google.colab  # pylint: disable=g-import-not-at-top,unused-import
  is_colab = True
except ImportError:
  is_colab = False


class LitWidget(object):
  """Class for using LIT inside notebooks."""

  def __init__(self, *args, height=1000, render=False,
               proxy_url=None, **kw):
    """Start LIT server and optionally render the UI immediately.

    Args:
      *args: Positional arguments for the LitApp.
      height: Height to display the LIT UI in pixels. Defaults to 1000.
      render: Whether to render the UI when this object is constructed.
          Defaults to False.
      proxy_url: Optional proxy URL, if using in a notebook with a server proxy.
          Defaults to None.
      **kw: Keyword arguments for the LitApp.
    """
    app_flags = server_flags.get_flags()
    app_flags['server_type'] = 'notebook'
    app_flags['host'] = 'localhost'
    app_flags['port'] = None
    app_flags['warm_start'] = 1
    app_flags.update(kw)

    lit_demo = dev_server.Server(
        *args, **app_flags)
    self._server = typing.cast(
        wsgi_serving.NotebookWsgiServer, lit_demo.serve())
    self._height = height
    self._proxy_url = proxy_url

    if render:
      self.render()

  def stop(self):
    """Stop the LIT server."""
    self._server.stop()

  def render(self, height=None):
    """Render the LIT UI in the output cell.

    Args:
      height: Optional height to display the LIT UI in pixels. If not specified,
          then the height specified in the constructor is used.
    """
    if not height:
      height = self._height
    if is_colab:
      _display_colab(self._server.port, height)
    else:
      _display_jupyter(self._server.port, height, self._proxy_url)


def _display_colab(port, height):
  """Display the LIT UI in colab.

  Args:
    port: The port the LIT server is running on.
    height: The height of the LIT UI in pixels.
  """

  shell = """
      (async () => {
          const url = new URL(
            await google.colab.kernel.proxyPort(%PORT%, {'cache': true}));
          const iframe = document.createElement('iframe');
          iframe.src = url;
          iframe.setAttribute('width', '100%');
          iframe.setAttribute('height', '%HEIGHT%px');
          iframe.setAttribute('frameborder', 0);
          document.body.appendChild(iframe);
      })();
  """
  replacements = [
      ('%PORT%', '%d' % port),
      ('%HEIGHT%', '%d' % height),
  ]
  for (k, v) in replacements:
    shell = shell.replace(k, v)

  script = display.Javascript(shell)
  display.display(script)


def _display_jupyter(port, height, proxy_url):
  """Display the LIT UI in colab.

  Args:
    port: The port the LIT server is running on.
    height: The height of the LIT UI in pixels.
    proxy_url: Optional proxy URL, if using in a notebook with a server proxy.
        If not provided, LIT also checks to see if the environment variable
        LIT_PROXY_URL is set, and if so, it uses that value as the proxy URL.
  """

  # Add height to jupyter output_scroll div to fully contain LIT UI.
  output_scroll_height = height + 10

  frame_id = 'lit-frame-{:08x}'.format(random.getrandbits(64))
  shell = """
    <style>div.output_scroll { height: %SCROLL_HEIGHT%px; }</style>
    <iframe id='%HTML_ID%' width='100%' height='%HEIGHT%' frameborder='0'>
    </iframe>
    <script>
      (function() {
        const frame = document.getElementById(%JSON_ID%);
        const url = new URL(%URL%, window.location);
        const port = %PORT%;
        if (port) {
          url.port = port;
        }
        frame.src = url;
      })();
    </script>
  """

  if proxy_url is None:
    proxy_url = os.environ.get('LIT_PROXY_URL')

  if proxy_url is not None:
    # Allow %PORT% in proxy_url.
    proxy_url = proxy_url.replace('%PORT%', '%d' % port)
    replacements = [
        ('%HTML_ID%', html.escape(frame_id, quote=True)),
        ('%JSON_ID%', json.dumps(frame_id)),
        ('%HEIGHT%', '%d' % height),
        ('%SCROLL_HEIGHT%', '%d' % output_scroll_height),
        ('%PORT%', '0'),
        ('%URL%', json.dumps(proxy_url)),
    ]
  else:
    replacements = [
        ('%HTML_ID%', html.escape(frame_id, quote=True)),
        ('%JSON_ID%', json.dumps(frame_id)),
        ('%HEIGHT%', '%d' % height),
        ('%SCROLL_HEIGHT%', '%d' % output_scroll_height),
        ('%PORT%', '%d' % port),
        ('%URL%', json.dumps('/')),
    ]

  for (k, v) in replacements:
    shell = shell.replace(k, v)

  iframe = display.HTML(shell)
  display.display(iframe)
