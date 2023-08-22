# Copyright 2023 Google LLC
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
"""LIT-specific replcement for transformers.file_utils.cached_path."""

from collections.abc import Mapping
import functools
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
from typing import BinaryIO, Optional
from urllib import parse as urllib_parse
import zipfile

# TODO(b/254110131): Reenable this import once the integration tests in the TODO
# below have been updated for compatibility.
# from absl import flags
from absl import logging
import filelock
import requests
import tqdm

_HTTP_SESSION = requests

DEFAULT_FILE_CACHE_PATH = '/tmp/lit_nlp/file_cache'
FILE_CACHE_PATH_ENV = os.getenv('LIT_FILE_CACHE_PATH', DEFAULT_FILE_CACHE_PATH)
# TODO(b/254110131): Update model initialization code in
# ablation_flip_int_test.py, hotflip_int_test.py, tcav_test.py, and
# thresholder_test.py to be compatible with the use of absl.flags here (via
# absl.testing.flagsaver, see Python Tips #51)
# FILE_CACHE_PATH_FLAG = flags.DEFINE_string(
#     name='lit_file_cache_path',
#     default=None,
#     help=f'Path to the file cache. Defaults to {DEFAULT_FILE_CACHE_PATH}.'
#     ' Overrides the LIT_FILE_CACHE_PATH environment variable. This path is'
#     ' expanded using os.expanduser(), see expected expansion behavior at'
#     ' https://docs.python.org/3/library/os.path.html#os.path.expanduser.',
# )


def _fetch_content(
    url: str,
    temp_file: BinaryIO,
    headers: Mapping[str, str],
    progress_indicator: Optional[tqdm.tqdm] = None,
    use_default_progress_indicator: bool = True,
) -> None:
  """Fetches HTTP content and writes it to the provided file."""
  head_response = _HTTP_SESSION.head(url, headers=headers, stream=True)
  head_response.raise_for_status()
  content_length = head_response.headers.get('Content-Length')
  total = int(content_length) if content_length is not None else None

  if progress_indicator is not None:
    progress = progress_indicator
  elif use_default_progress_indicator:
    progress = tqdm.tqdm(
        unit='B',
        unit_scale=True,
        total=total,
        initial=0,
        desc='Downloading',
    )
  else:
    progress = None

  get_response = _HTTP_SESSION.get(url=url, stream=True, headers=headers)
  get_response.raise_for_status()
  for chunk in get_response.iter_content(chunk_size=1024):
    if chunk:
      if progress is not None:
        progress.update(len(chunk))
      temp_file.write(chunk)

  if progress is not None:
    progress.close()


def _get_extacted_dir(output_path: str) -> str:
  """Extracts and returns the directory containing the provided archive."""
  is_zip = zipfile.is_zipfile(output_path)
  if not (is_zip or tarfile.is_tarfile(output_path)):
    return output_path

  output_dir, output_file = os.path.split(output_path)
  output_extracted_dir_name = output_file.replace('.', '-') + '-extracted'
  output_extracted_path = os.path.join(output_dir, output_extracted_dir_name)

  if os.path.isdir(output_extracted_path) and os.listdir(output_extracted_path):
    return output_extracted_path

  lock_path = output_path + '.lock'
  with filelock.FileLock(lock_path):
    shutil.rmtree(output_extracted_path, ignore_errors=True)
    os.makedirs(output_extracted_path)

    if is_zip:
      with zipfile.ZipFile(output_path, 'r') as zip_file:
        zip_file.extractall(output_extracted_path)
        zip_file.close()
    else:
      tar_file = tarfile.open(output_path)
      tar_file.extractall(output_extracted_path)
      tar_file.close()

  return output_extracted_path


def _get_from_cache(
    url: str,
    local_files_only: bool = False,
    progress_indicator: Optional[tqdm.tqdm] = None,
) -> str:
  """Downloads and returns the cache path to the content at the provided URL."""
  # TODO(b/254110131): Update to use FILE_CACHE_PATH_FLAG.value once the
  # integration tests in the TODO above have been updated for compatibility.
  lit_cache_dir = os.path.expanduser(FILE_CACHE_PATH_ENV)
  headers = {'user-agent': f'lit_nlp; python/{sys.version.split()[0]}'}
  etag: Optional[str] = None
  url_to_download: str = url
  os.makedirs(lit_cache_dir, exist_ok=True)

  # Attempt to get the ETag for this resource.
  if not local_files_only:
    try:
      head_response = _HTTP_SESSION.head(
          url, headers=headers, stream=True, allow_redirects=False
      )
      head_response.raise_for_status()
      etag = str(
          head_response.headers.get('X-Linked-Etag')
          or head_response.headers.get('ETag')
      )
      if etag is None:
        raise OSError(
            'No ETag found for resource, reproducibility will be unreliable.'
        )

      # If the HEAD responds with a redirect, download from that URL if the
      # content is not in the cache, but use the original URL for the filename.
      # Otherwise, download from the original URL.
      if 300 <= head_response.status_code < 400:
        url_to_download = str(head_response.headers['Location'])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
      pass  # etag is already None, url_to_download is already set.

  filename = filename_fom_url(url, etag=etag)
  cache_path = os.path.join(lit_cache_dir, filename)

  if os.path.exists(cache_path):
    logging.info('File %s exists in cache as %s', url, cache_path)
    return cache_path
  elif etag is None:
    raise ValueError(
        'Encountered a connection error and cannot find the requested files in'
        ' the cached path. Please try again and ensure your Internet connection'
        ' is on.'
    )

  lock_path = cache_path + '.lock'
  with filelock.FileLock(lock_path):
    if os.path.exists(cache_path):  # Download completed while activating lock.
      logging.info('File %s existing in cache as %s', url, cache_path)
      return cache_path  # Returning early will release the lock.

    temp_file_manager = functools.partial(
        tempfile.NamedTemporaryFile, mode='wb', dir=lit_cache_dir, delete=False
    )
    with temp_file_manager() as temp_file:
      logging.info('%s not found in cache.', url)
      _fetch_content(
          url_to_download,
          temp_file,
          headers=headers,
          progress_indicator=progress_indicator
      )
      logging.info('%s downloaded to %s', url, temp_file.name)

    logging.info('Storing %s in cache at %s', url, cache_path)
    os.replace(temp_file.name, cache_path)

  return cache_path


def filename_fom_url(url: str, etag: Optional[str] = None) -> str:
  """Converts a URL, and optional etag, to a filename using a SHA 256 hash."""
  url_as_bytes = url.encode('utf-8')
  filename = hashlib.sha256(url_as_bytes).hexdigest()[:16]

  if etag:
    etag_bytes = etag.encode('utf-8')
    etag_sha = hashlib.sha256(etag_bytes).hexdigest()[:8]
    filename += f'_{etag_sha}'

  # Preserve file extension for URLs that include it.
  parsed_url = urllib_parse.urlparse(url)
  url_path = parsed_url.path
  _, extension = os.path.splitext(url_path)
  if extension:
    filename += extension

  return filename


def is_remote(url_of_filepath: str) -> bool:
  parsed = urllib_parse.urlparse(url_of_filepath)
  return parsed.scheme in ('http', 'https')


def cached_path(
    url_or_filepath: str,
    extract_compressed_file: bool = False,
    local_files_only: bool = False,
    progress_indicator: Optional[tqdm.tqdm] = None,
) -> str:
  """Get the path to the locally cached resource."""
  if is_remote(url_or_filepath):
    logging.info('%s is remote', url_or_filepath)
    output_path = _get_from_cache(
        url_or_filepath,
        local_files_only=local_files_only,
        progress_indicator=progress_indicator,
    )
  elif os.path.exists(url_or_filepath):
    logging.info('%s is local', url_or_filepath)
    output_path = url_or_filepath
  elif not urllib_parse.urlparse(url_or_filepath).scheme:
    raise EnvironmentError(f'File not found: {url_or_filepath}')
  else:
    raise ValueError(
        f'Unable to parse as URL or local path: {url_or_filepath}'
    )

  if not extract_compressed_file:
    return output_path
  else:
    return _get_extacted_dir(output_path)
