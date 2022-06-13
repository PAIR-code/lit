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

import 'jasmine';

// These are needed to preserve the import, as they are referenced indirectly.
// through the names of elements they define.
// tslint:disable-next-line:no-unused-variable
import {ClassificationModule} from '../modules/classification_module';
// tslint:disable-next-line:no-unused-variable
import {DatapointEditorModule} from '../modules/datapoint_editor_module';

import {canonicalizeLayout, formatForDisplay, LitCanonicalLayout, LitComponentLayout} from './types';

describe('formatForDisplay test', () => {

  it('formats string for display', () => {
    const testString = 'test';
    const formatted = formatForDisplay(testString);
    expect(formatted).toBe(testString);
  });

  it('formats number for display', () => {
    const formatted = formatForDisplay(1.23456789);
    expect(formatted).toBe('1.2346');
  });

  it('formats boolean for display', () => {
    let formatted = formatForDisplay(true);
    expect(formatted).toBe('✔');

    formatted = formatForDisplay(false);
    expect(formatted).toBe(' ');
  });

  it('formats array for display', () => {
    // number array.
    let formatted = formatForDisplay([1.23456789, 2.3456789]);
    expect(formatted).toBe('1.2346, 2.3457');

    // number|string array.
    formatted = formatForDisplay(['a', 1.23456789, 2.3456789]);
    expect(formatted).toBe('a, 1.2346, 2.3457');

    // string array
    formatted = formatForDisplay(['a', 'b', 'cd']);
    expect(formatted).toBe('a, b, cd');
  });
});

const MOCK_LAYOUT: LitComponentLayout = {
  components: {
    'Main': [
      'datapoint-editor-module',
    ],
    'internals': [
      // Duplicated per model and in compareDatapoints mode.
      'classification-module',
    ],
  },
  layoutSettings: {hideToolbar: true, mainHeight: 90, centerPage: true},
  description: 'Mock layout for testing.'
};

const CANONICAL_MOCK_LAYOUT: LitCanonicalLayout = {
  upper: {
    'Main': [
      'datapoint-editor-module',
    ],
  },
  lower: {
    'internals': [
      // Duplicated per model and in compareDatapoints mode.
      'classification-module',
    ],
  },
  layoutSettings: {hideToolbar: true, mainHeight: 90, centerPage: true},
  description: 'Mock layout for testing.'
};

describe('canonicalizeLayout test', () => {
  it('correctly converts a legacy layout', () => {
    const converted = canonicalizeLayout(MOCK_LAYOUT);
    expect(converted).toEqual(CANONICAL_MOCK_LAYOUT);
  });

  it('does not modify an already canonical layout', () => {
    const converted = canonicalizeLayout(CANONICAL_MOCK_LAYOUT);
    expect(converted).toEqual(CANONICAL_MOCK_LAYOUT);
  });
});
