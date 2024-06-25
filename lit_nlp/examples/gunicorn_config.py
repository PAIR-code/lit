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
"""gunicorn configuration for cloud-hosted demos."""

import os

_DEMO_NAME = os.getenv('DEMO_NAME', 'glue')
_DEMO_PORT = os.getenv('DEMO_PORT', '5432')

bind = f'0.0.0.0:{_DEMO_PORT}'
timeout = 3600
threads = 8
worker_class = 'gthread'
wsgi_app = f'lit_nlp.examples.{_DEMO_NAME}.demo:get_wsgi_app()'
