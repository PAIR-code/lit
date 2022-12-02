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
 * A special webpack loader that creates a JS module that exports a CSS file
 * as a lit-html css template string result named "styles", to match internal
 * google lit-html css preprocessing.
 */
module.exports = function loader(source) {
  return `
    import {css} from 'lit';
    export const styles = css\`${ source }\`;
  `;
}

