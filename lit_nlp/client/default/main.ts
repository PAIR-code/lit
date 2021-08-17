/**
 * @license
 * Copyright 2020 Google LLC
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

/**
 * The main entry point for the LIT App, compiled into `bundle.js`. This file
 * imports the LIT App container web component, a declared layout of LIT
 * Modules, and the LIT App core service, then initializes the app.
 */

// Imports the main LIT App web component, which is declared here then attached
// to the DOM as <lit-app>
import '../core/lit_app';

import {app} from '../core/app';

import {LAYOUTS} from './layout';

// Initialize the app core logic, using the specified declared layout.
app.initialize(LAYOUTS);
