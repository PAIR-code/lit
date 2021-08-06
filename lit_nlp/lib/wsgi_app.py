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
"""Simple WSGI app implementation.

This takes a list of handlers, and creates a WSGI application that can be served
through a variety of methods.

Why not use Flask or something? Historical reasons, and if it ain't broke, don't
fix it.
"""

import mimetypes
import os
import time
import traceback
import wsgiref.handlers

from absl import logging
import six
from six.moves.urllib.parse import urlparse
from werkzeug import wrappers


def _LoadResource(path):
  """Load the resource at given path.

  Args:
    path: a string resource path.

  Returns:
    The contents of that resource.

  Raises:
    ValueError: If the path is not set up correctly.
    IOError: If the path is not found, or the resource can't be opened.
  """

  try:
    with open(path, 'rb') as f:
      return f.read()
  except IOError as e:
    logging.warning('IOError %s on path %s', e, path)
    raise e


class App(object):
  """Standalone WSGI app that can serve files, etc."""

  _TEXTUAL_MIMETYPES = set([
      'application/javascript',
      'application/json',
      'application/json+protobuf',
      'image/svg+xml',
      'text/css',
      'text/csv',
      'text/html',
      'text/json',
      'text/plain',
      'text/tab-separated-values',
      'text/x-protobuf',
  ])

  def __init__(self, handlers, project_root, index_file='index.html'):
    self._handlers = handlers
    self._project_root = project_root
    self._index_file = index_file

  def respond(  # pylint: disable=invalid-name
      self,
      request,
      content,
      content_type,
      code=200,
      expires=0,
      content_encoding=None):
    """Construct a werkzeug WSGI response object.

    Args:
      request: A werkzeug Request object. Used mostly to check the
        Accept-Encoding header.
      content: Payload data as bytes or unicode string (will be UTF-8 encoded).
      content_type: Media type only - "charset=utf-8" will be added for text.
      code: Numeric HTTP status code to use.
      expires: Second duration for browser caching, default 0.
      content_encoding: Encoding if content is already encoded, e.g. 'gzip'.

    Returns:
      A werkzeug Response object (a WSGI application).
    """
    if isinstance(content, six.text_type):
      content = content.encode('utf-8')
    if content_type in self._TEXTUAL_MIMETYPES:
      content_type += '; charset=utf-8'
    headers = []
    headers.append(('Content-Length', str(len(content))))
    if content_encoding:
      headers.append(('Content-Encoding', content_encoding))
    if expires > 0:
      e = wsgiref.handlers.format_date_time(time.time() + float(expires))
      headers.append(('Expires', e))
      headers.append(('Cache-Control', 'private, max-age=%d' % expires))
    else:
      headers.append(('Expires', '0'))
      headers.append(('Cache-Control', 'no-cache, must-revalidate'))
    if request.method == 'HEAD':
      content = None
    return wrappers.Response(
        response=content,
        status=code,
        headers=headers,
        content_type=content_type)

  def _ServeStaticFile(self, request, path):
    """Serves the static file located at the given path.

    Args:
      request: A Werkzeug Request object.
      path: The path of the static file, relative to the current directory.

    Returns:
      A Werkzeug Response object.
    """
    if not self._PathIsSafe(path):
      logging.info('path %s not safe, sending 400', path)
      # Traversal attack, so 400.
      return self.respond(request, 'Path not safe', 'text/plain', 400)

    # Open the file and read it.
    try:
      contents = _LoadResource(path)
    except IOError:
      logging.info('path %s not found, sending 404', path)
      return self.respond(request, 'Not found', 'text/plain', code=404)

    mimetype, content_encoding = mimetypes.guess_type(path)
    mimetype = mimetype or 'application/octet-stream'
    return self.respond(
        request,
        contents,
        mimetype,
        expires=3600,
        content_encoding=content_encoding)

  def _PathIsSafe(self, path):
    """Check path is safe (stays within current directory).

    This is for preventing directory-traversal attacks.

    Args:
      path: The path to check for safety.

    Returns:
      True if the given path stays within the project directory, and false
      if it would escape to a higher directory. E.g. _path_is_safe('index.html')
      returns true, but _path_is_safe('../../../etc/password') returns false.
    """
    base = os.path.abspath(self._project_root)
    absolute_path = os.path.abspath(path)
    prefix = os.path.commonprefix([base, absolute_path])
    return prefix == base

  def _ServeCustomHandler(self, request, clean_path, environ):
    return self._handlers[clean_path](self, request, environ)

  def __call__(self, environ, start_response):
    """Implementation of the WSGI interface."""
    request = wrappers.Request(environ)

    try:
      parsed_url = urlparse(request.path)

      # Remove a trailing slash, if present.
      clean_path = parsed_url.path
      if clean_path.endswith('/'):
        clean_path = clean_path[:-1]

      if clean_path in self._handlers:
        return self._ServeCustomHandler(request, clean_path, environ)(
            environ, start_response)
      else:
        is_index = not clean_path or clean_path == '/index.html'
        if is_index:
          clean_path = os.path.join(self._project_root, self._index_file)
        else:
          # Strip off the leading forward slash. Don't do it for index because
          # in the vulcanized version we use an absolute path.
          clean_path = os.path.join(self._project_root, clean_path.lstrip('/'))

        response = self._ServeStaticFile(request, clean_path)

    except Exception as e:  # pylint: disable=broad-except
      errors = (str(e), str(traceback.format_exc()))
      html_response = (
          'Uncaught error: %s\n\nDetails: %s' % errors)
      logging.error('Uncaught error: %s \n\n %s', *errors)
      response = self.respond(request, html_response, 'text/html', 500)

    return response(environ, start_response)
