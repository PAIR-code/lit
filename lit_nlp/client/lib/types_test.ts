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

  const tests: Array<[string, unknown, string|number]> = [
    ['boolean literal true', true, '✔'],
    ['boolean literal false', false, ' '],
    ['number', 1.234567, 1.235],
    ['number[]', [1.23456789, 2.3456789], '1.235, 2.346'],
    ['scored text candidate w/ score', ['text', 2.3456789], 'text (2.346)'],
    ['scored text candidate no score', ['text', null], 'text'],
    ['scored text candidates', [['1st', 1], ['2nd', 2]], '1st (1)\n\n2nd (2)'],
    ['string', 'test', 'test'],
    ['string[]', ['a', 'b', 'cd'], 'a, b, cd'],
    ['Array<string | number>', ['a', 1.23456789, 2.3456789], 'a, 1.235, 2.346']
  ];

  tests.forEach(([name, value, expected]: [string, unknown, string|number]) => {
    it(`formats ${name} for display`, () => {
      expect(formatForDisplay(value)).toBe(expected);
    });
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

const CANONICAL_LAYOUT: LitCanonicalLayout = {
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
  type LitLayout = LitComponentLayout|LitCanonicalLayout;
  const tests: Array<[string, LitLayout, LitCanonicalLayout]> = [
    ['converts a legacy layout', MOCK_LAYOUT, CANONICAL_LAYOUT],
    ['leaves canonical layouts alone', CANONICAL_LAYOUT,  CANONICAL_LAYOUT],
  ];

  tests.forEach(([name, layout, expected]) => {
    it(name, () => {expect(canonicalizeLayout(layout)).toEqual(expected);});
  });
});
