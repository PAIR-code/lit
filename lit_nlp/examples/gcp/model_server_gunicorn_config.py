# Copyright 2024 Google LLC
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

_PORT = os.getenv('PORT', '8080')

bind = f'0.0.0.0:{_PORT}'
timeout = 3600
threads = 8
worker_class = 'gthread'
wsgi_app = 'lit_nlp.examples.gcp.model_server:get_wsgi_app()'
