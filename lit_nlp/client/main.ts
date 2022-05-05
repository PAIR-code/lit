/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {app} from './core/app';

// Initialize the app here so that this is not automatically called in unit
// tests. Otherwise, we would need to inject mocks into the global singleton
// so that it doesn't crash when the test is run without a backend server.
// TODO(lit-dev): consider using DOMContentLoaded to initialize sooner?
window.addEventListener('load', () => {
  app.initialize();
});
