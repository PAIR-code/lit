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
"""Tests for cf_utils."""

from absl.testing import absltest
from lit_nlp.components import cf_utils


class CfUtilsTest(absltest.TestCase):

  def test_tokenize_url(self):
    url = "http://www.google.com"
    tokens = ["http", "www", "google", "com"]
    self.assertEqual(tokens, cf_utils.tokenize_url(url))

    url = "http://www.gmail.com/mail/u/1/#inbox"
    tokens = ["http", "www", "gmail", "com", "mail", "u", "1", "inbox"]
    self.assertEqual(tokens, cf_utils.tokenize_url(url))

    url = "foobar"
    tokens = ["foobar"]
    self.assertEqual(tokens, cf_utils.tokenize_url(url))

    url = ""
    self.assertEmpty(cf_utils.tokenize_url(url))

    url = "/"
    self.assertEmpty(cf_utils.tokenize_url(url))

    url = "//"
    self.assertEmpty(cf_utils.tokenize_url(url))

    url = "//foobar/"
    tokens = ["foobar"]
    self.assertEqual(tokens, cf_utils.tokenize_url(url))

  def test_ablate_url_tokens(self):
    url = "http://www.gmail.com/mail/u/1/#inbox"
    token_idxs_to_ablate = [0]
    expected_url = "://www.gmail.com/mail/u/1/#inbox"
    self.assertEqual(expected_url,
                     cf_utils.ablate_url_tokens(url, token_idxs_to_ablate))

    token_idxs_to_ablate = [0, 3, 4]
    expected_url = "://www.gmail.//u/1/#inbox"
    self.assertEqual(expected_url,
                     cf_utils.ablate_url_tokens(url, token_idxs_to_ablate))

    token_idxs_to_ablate = [4, 3, 0]
    expected_url = "://www.gmail.//u/1/#inbox"
    self.assertEqual(expected_url,
                     cf_utils.ablate_url_tokens(url, token_idxs_to_ablate))

    token_idxs_to_ablate = [0, 1, 2, 3, 4, 5, 6, 7]
    expected_url = "://..////#"
    self.assertEqual(expected_url,
                     cf_utils.ablate_url_tokens(url, token_idxs_to_ablate))

    token_idxs_to_ablate = []
    self.assertEqual(url,
                     cf_utils.ablate_url_tokens(url, token_idxs_to_ablate))


if __name__ == "__main__":
  absltest.main()
