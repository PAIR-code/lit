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
