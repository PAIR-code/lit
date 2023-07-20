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

import {formatForDisplay} from './types';

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
