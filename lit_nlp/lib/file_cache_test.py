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

import io
import os
import tarfile
import zipfile

from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.lib import file_cache


class FileCacheTest(parameterized.TestCase):

  # ETag can have strong (an ASCII character string) or weak (prefixed by 'W/)
  # validation. See MDN for more:
  # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag#directives
  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          etag='',
          expected='49e48471af489fc1.chkpt',
      ),
      dict(
          testcase_name='standard_validator',
          etag='a2c4e67',
          expected='49e48471af489fc1_043d7490.chkpt',
      ),
      dict(
          testcase_name='weak_validator',
          etag='W/a2c4e67',
          expected='49e48471af489fc1_9fee3ee2.chkpt',
      ),
  )
  def test_filename_fom_url_etag(self, etag: str, expected: str):
    url = 'https://example.com/testdata/model.chkpt'
    filename = file_cache.filename_fom_url(url, etag)
    self.assertEqual(filename, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          url='',
          expected='e3b0c44298fc1c14',
      ),
      dict(
          testcase_name='extensionless',
          url='https://example.com/testdata/model',
          expected='adb48b9e4d4f2dfa',
      ),
      dict(
          testcase_name='HDF5_file',
          url='https://example.com/testdata/model.h5',
          expected='c56165097a137459.h5',
      ),
      dict(
          testcase_name='JSON_file',
          url='https://example.com/testdata/model.json',
          expected='56d84f3bc9b95492.json',
      ),
      dict(
          testcase_name='Tar_archive',
          url='https://example.com/testdata/model.tar',
          expected='5882fbfd78d7abb8.tar',
      ),
      dict(
          testcase_name='Zip_archive',
          url='https://example.com/testdata/model.zip',
          expected='94797e8635299d8a.zip',
      ),
  )
  def test_filename_fom_url_no_etag(self, url: str, expected: str):
    filename = file_cache.filename_fom_url(url)
    self.assertEqual(filename, expected)

  @parameterized.named_parameters(
      ('empty', '', False),
      ('Amazon_S3', 's3://testdata/model.chkpt', False),
      ('FTP', 'ftp://testdata/model.chkpt', False),
      ('Google_Cloud_Storage', 'gs://testdata/model.chkpt', False),
      ('HTTP', 'http://example.com/testdata/model.chkpt', True),
      ('HTTPS', 'https://example.com/testdata/model.chkpt', True),
      ('local_file', '/usr/local/testdata/model.chkpt', False),
  )
  def test_is_remote(self, url: str, expected: bool):
    is_remote = file_cache.is_remote(url)
    self.assertEqual(is_remote, expected)

  def test_cached_path_extracts_valid_tar(self):
    temp_dir = self.create_tempdir().full_path
    tar_path = os.path.join(temp_dir, 'model.tar.gz')
    payload = b'{"model": "ok"}'
    with tarfile.open(tar_path, 'w:gz') as tar:
      info = tarfile.TarInfo(name='subdir/config.json')
      info.size = len(payload)
      tar.addfile(info, io.BytesIO(payload))

    extracted_dir = file_cache.cached_path(
        tar_path, extract_compressed_file=True
    )
    extracted_file = os.path.join(extracted_dir, 'subdir', 'config.json')
    self.assertTrue(os.path.isfile(extracted_file))
    with open(extracted_file, 'rb') as f:
      self.assertEqual(f.read(), payload)

  def test_cached_path_extracts_valid_zip(self):
    temp_dir = self.create_tempdir().full_path
    zip_path = os.path.join(temp_dir, 'model.zip')
    payload = b'{"model": "ok"}'
    with zipfile.ZipFile(zip_path, 'w') as zf:
      zf.writestr('subdir/config.json', payload)

    extracted_dir = file_cache.cached_path(
        zip_path, extract_compressed_file=True
    )
    extracted_file = os.path.join(extracted_dir, 'subdir', 'config.json')
    self.assertTrue(os.path.isfile(extracted_file))
    with open(extracted_file, 'rb') as f:
      self.assertEqual(f.read(), payload)

  @parameterized.named_parameters(
      ('parent_traversal', '../escaped.txt'),
      ('nested_parent_traversal', 'subdir/../../escaped.txt'),
      ('absolute_parent_traversal', '/../escaped.txt'),
  )
  def test_cached_path_rejects_tar_traversal(self, malicious_name: str):
    temp_dir = self.create_tempdir().full_path
    archive_dir = os.path.join(temp_dir, 'cache')
    os.makedirs(archive_dir)
    tar_path = os.path.join(archive_dir, 'malicious.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
      valid_info = tarfile.TarInfo(name='valid.txt')
      valid_info.size = 2
      tar.addfile(valid_info, io.BytesIO(b'ok'))
      bad_info = tarfile.TarInfo(name=malicious_name)
      bad_info.size = 4
      tar.addfile(bad_info, io.BytesIO(b'evil'))

    with self.assertRaises(tarfile.FilterError):
      file_cache.cached_path(tar_path, extract_compressed_file=True)

    self.assertFalse(os.path.exists(os.path.join(temp_dir, 'escaped.txt')))
    self.assertFalse(os.path.exists(os.path.join(archive_dir, 'escaped.txt')))
    expected_extracted = os.path.join(archive_dir, 'malicious-tar-gz-extracted')
    self.assertFalse(os.path.exists(expected_extracted))

  def test_cached_path_sanitizes_tar_absolute_path(self):
    temp_dir = self.create_tempdir().full_path
    archive_dir = os.path.join(temp_dir, 'cache')
    os.makedirs(archive_dir)
    outside_target = os.path.join(temp_dir, 'outside_abs.txt')
    tar_path = os.path.join(archive_dir, 'abs_path.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
      abs_info = tarfile.TarInfo(name=outside_target)
      abs_info.size = 4
      tar.addfile(abs_info, io.BytesIO(b'safe'))

    extracted_dir = file_cache.cached_path(
        tar_path, extract_compressed_file=True
    )
    self.assertFalse(os.path.exists(outside_target))
    self.assertTrue(
        os.path.isfile(os.path.join(extracted_dir, outside_target.lstrip('/')))
    )

  def test_cached_path_rejects_tar_external_symlink(self):
    temp_dir = self.create_tempdir().full_path
    archive_dir = os.path.join(temp_dir, 'cache')
    os.makedirs(archive_dir)
    tar_path = os.path.join(archive_dir, 'symlink.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
      sym_info = tarfile.TarInfo(name='link_out')
      sym_info.type = tarfile.SYMTYPE
      sym_info.linkname = '../../outside_target'
      tar.addfile(sym_info)

    with self.assertRaises(tarfile.FilterError):
      file_cache.cached_path(tar_path, extract_compressed_file=True)

    expected_extracted = os.path.join(archive_dir, 'symlink-tar-gz-extracted')
    self.assertFalse(os.path.exists(expected_extracted))

  @parameterized.named_parameters(
      ('parent_traversal', '../escaped.txt'),
      ('nested_parent_traversal', 'subdir/../../escaped.txt'),
      ('absolute_path', '/tmp/escaped_abs.txt'),
  )
  def test_cached_path_rejects_zip_traversal(self, malicious_name: str):
    temp_dir = self.create_tempdir().full_path
    archive_dir = os.path.join(temp_dir, 'cache')
    os.makedirs(archive_dir)
    zip_path = os.path.join(archive_dir, 'malicious.zip')
    with zipfile.ZipFile(zip_path, 'w') as zf:
      zf.writestr('valid.txt', b'ok')
      zf.writestr(malicious_name, b'evil')

    with self.assertRaises(ValueError):
      file_cache.cached_path(zip_path, extract_compressed_file=True)

    self.assertFalse(os.path.exists(os.path.join(temp_dir, 'escaped.txt')))
    self.assertFalse(os.path.exists(os.path.join(archive_dir, 'escaped.txt')))
    expected_extracted = os.path.join(archive_dir, 'malicious-zip-extracted')
    self.assertFalse(os.path.exists(expected_extracted))

  # TODO(b/285157349, b/254110131): Add UT/ITs for file_cache.cached_path().
  # Conditions should include:
  #
  # * File paths with lit_file_cache_path flag is set (UT).
  # * File paths with LIT_FILE_CACHE_PATH env var set (UT).
  # * File paths with lit_file_cache_path flag and LIT_FILE_CACHE_PATH env var
  #   are set (UT; flag should win).
  # * Local paths that include a file extension (UT)
  # * Local paths that do not include a file extension (UT)
  # * Local paths for TAR and Zip archives in the cache (UT)
  # * Local paths for TAR and Zip archives not in the cache (IT)
  # * URLs in cache for paths that include a file extension (UT)
  # * URLs in cache for paths that do not include a file extension (UT)
  # * URls in cache for TAR and Zip archives (UT)
  # * URLs not in cache for paths that include a file extension (IT?)
  # * URLs not in cache for paths that do not include a file extension (IT?)
  # * URls not in cache for TAR and Zip archives (IT?)


if __name__ == '__main__':
  absltest.main()
