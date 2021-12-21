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
from typing import cast, List, Optional
import urllib.parse
import attr
# pytype: disable=import-error
from IPython import display
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import dtypes
from lit_nlp.lib import wsgi_serving

try:
  import google.colab  # pylint: disable=g-import-not-at-top,unused-import
  is_colab = True
except ImportError:
  is_colab = False

MODEL_PREDS_MODULES = [
    'span-graph-gold-module-vertical',
    'span-graph-module-vertical',
    'classification-module',
    'multilabel-module',
    'regression-module',
    'lm-prediction-module',
    'generated-text-module',
    'annotated-text-gold-module',
    'annotated-text-module',
    'generated-image-module',
]

LIT_NOTEBOOK_LAYOUT = dtypes.LitCanonicalLayout(
    upper={
        'Predictions': ['simple-data-table-module'] + MODEL_PREDS_MODULES,
        'Explanations': ['simple-datapoint-editor-module'] +
                        MODEL_PREDS_MODULES + [
                            'salience-map-module', 'sequence-salience-module',
                            'attention-module'
                        ],
        'Analysis':
            ['metrics-module', 'confusion-matrix-module', 'scalar-module'],
    })


@attr.s(auto_attribs=True, kw_only=True)
class RenderConfig(object):
  """Config options for widget rendering."""
  tab: Optional[str] = None
  upper_tab: Optional[str] = None
  layout: Optional[str] = None
  dataset: Optional[str] = None
  models: Optional[List[str]] = None

  def get_query_str(self):
    """Convert config object to query string for LIT URL."""
    def _encode(v):
      if isinstance(v, (list, tuple)):
        return ','.join(v)
      return v

    string_params = {
        k: _encode(v) for k, v in attr.asdict(self).items() if v is not None}
    return '?' + urllib.parse.urlencode(string_params)


class LitWidget(object):
  """Class for using LIT inside notebooks."""

  def __init__(self,
               *args,
               height=1000,
               render=False,
               proxy_url=None,
               layouts: Optional[dtypes.LitComponentLayouts] = None,
               **kw):
    """Start LIT server and optionally render the UI immediately.

    Args:
      *args: Positional arguments for the LitApp.
      height: Height to display the LIT UI in pixels. Defaults to 1000.
      render: Whether to render the UI when this object is constructed. Defaults
        to False.
      proxy_url: Optional proxy URL, if using in a notebook with a server proxy.
        Defaults to None.
      layouts: Optional custom UI layouts. TODO(lit-dev): support simple module
        lists here as well.
      **kw: Keyword arguments for the LitApp.
    """
    app_flags = server_flags.get_flags()
    app_flags['server_type'] = 'notebook'
    app_flags['host'] = 'localhost'
    app_flags['port'] = None
    app_flags['warm_start'] = 1
    layouts = dict(layouts or {})
    if 'notebook' not in layouts:
      layouts['notebook'] = LIT_NOTEBOOK_LAYOUT
    # This will be 'notebook' unless custom layouts are also given in Python.
    app_flags['default_layout'] = list(layouts.keys())[0]
    app_flags.update(kw)

    lit_demo = dev_server.Server(*args, layouts=layouts, **app_flags)
    self._server = cast(wsgi_serving.NotebookWsgiServer, lit_demo.serve())
    self._height = height
    self._proxy_url = proxy_url

    if render:
      self.render()

  def stop(self):
    """Stop the LIT server."""
    self._server.stop()

  def render(self, height=None, open_in_new_tab=False,
             ui_params: Optional[RenderConfig] = None):
    """Render the LIT UI in the output cell.

    Args:
      height: Optional height to display the LIT UI in pixels. If not specified,
          then the height specified in the constructor is used.
      open_in_new_tab: Whether to show the UI in a new tab instead of in the
        output cell. Defaults to false.
      ui_params: Optional configuration options for the LIT UI's state.
    """
    if not height:
      height = self._height
    if not ui_params:
      ui_params = RenderConfig()
    if is_colab:
      _display_colab(self._server.port, height, open_in_new_tab, ui_params)
    else:
      _display_jupyter(self._server.port, height, self._proxy_url,
                       open_in_new_tab, ui_params)


def _display_colab(port, height, open_in_new_tab, ui_params: RenderConfig):
  """Display the LIT UI in colab.

  Args:
    port: The port the LIT server is running on.
    height: The height of the LIT UI in pixels.
    open_in_new_tab: Whether to show the UI in a new tab instead of in the
      output cell.
    ui_params: RenderConfig of options for the LIT UI.
  """

  params = ui_params.get_query_str()

  if open_in_new_tab:
    shell = """
      (async () => {
          const proxyPort = await google.colab.kernel.proxyPort(
            %PORT%, {'cache': true})
          const url = new URL(proxyPort + '%PARAMS%')
          const a = document.createElement('a');
          a.href = "javascript:void(0);"
          a.onclick = (e) => window.open(url, "_blank");
          a.innerHTML = url;
          document.body.appendChild(a);
          window.open(url, "_blank");
      })();
    """
  else:
    shell = """
      (async () => {
          const proxyPort = await google.colab.kernel.proxyPort(
            %PORT%, {'cache': true})
          const url = new URL(proxyPort + '%PARAMS%')
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
      ('%PARAMS%', '%s' % params),
  ]
  for (k, v) in replacements:
    shell = shell.replace(k, v)

  script = display.Javascript(shell)
  display.display(script)


def _display_jupyter(port, height, proxy_url, open_in_new_tab,
                     ui_params: RenderConfig):
  """Display the LIT UI in jupyter.

  Args:
    port: The port the LIT server is running on.
    height: The height of the LIT UI in pixels.
    proxy_url: Optional proxy URL, if using in a notebook with a server proxy.
        If not provided, LIT also checks to see if the environment variable
        LIT_PROXY_URL is set, and if so, it uses that value as the proxy URL.
    open_in_new_tab: Whether to show the UI in a new tab instead of in the
      output cell.
    ui_params: RenderConfig of options for the LIT UI.
  """

  # Add height to jupyter output_scroll div to fully contain LIT UI.
  output_scroll_height = height + 10

  params = ui_params.get_query_str()

  frame_id = 'lit-frame-{:08x}'.format(random.getrandbits(64))
  if open_in_new_tab:
    shell = """
      <a href="javascript:void(0);" id="%HTML_ID%"></a>
      <script>
        (function() {
          const urlStr = %URL% + '%PARAMS%'
          const url = new URL(urlStr, window.location);
          const port = %PORT%;
          if (port) {
            url.port = port;
          }
          const a = document.getElementById(%JSON_ID%);
          a.innerHTML = url;
          a.onclick = (e) => window.open(url, "_blank");
          window.open(url, "_blank");
        })();
      </script>
    """
  else:
    shell = """
      <style>div.output_scroll { height: %SCROLL_HEIGHT%px; }</style>
      <iframe id='%HTML_ID%' width='100%' height='%HEIGHT%' frameborder='0'>
      </iframe>
      <script>
        (function() {
          const frame = document.getElementById(%JSON_ID%);
          const urlStr = %URL% + '%PARAMS%'
          const url = new URL(urlStr, window.location);
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
        ('%PARAMS%', '%s' % params),
    ]
  else:
    replacements = [
        ('%HTML_ID%', html.escape(frame_id, quote=True)),
        ('%JSON_ID%', json.dumps(frame_id)),
        ('%HEIGHT%', '%d' % height),
        ('%SCROLL_HEIGHT%', '%d' % output_scroll_height),
        ('%PORT%', '%d' % port),
        ('%URL%', json.dumps('/')),
        ('%PARAMS%', '%s' % params),
    ]

  for (k, v) in replacements:
    shell = shell.replace(k, v)

  iframe = display.HTML(shell)
  display.display(iframe)
