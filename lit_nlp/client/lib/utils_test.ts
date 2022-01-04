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
 * Testing for utils.ts
 */

import 'jasmine';
import {Spec} from '../lib/types';
import * as utils from './utils';

describe('randInt test', () => {
  it('generates random integers in a given range', async () => {
    let start = 1;
    let end = 5;
    let result = utils.randInt(start, end);
    expect(result).not.toBeLessThan(start);
    expect(result).toBeLessThan(end);

    start = -2;
    end = 2;
    result = utils.randInt(start, end);
    expect(result).not.toBeLessThan(start);
    expect(result).toBeLessThan(end);
  });

  it('generates random integers when start and end are equal', async () => {
    const start = 1;
    const end = 1;
    const result = utils.randInt(start, end);
    expect(result).toBe(1);
  });
});

describe('setEquals test', () => {
  it('correctly determines that empty sets are equal', async () => {
    const emptyA = new Set([]);
    const emptyB = new Set([]);
    expect(utils.setEquals(emptyA, emptyB)).toBe(true);
  });

  it('correctly determines that sets are equal', async () => {
    let a = new Set(['a']);
    let b = new Set(['a']);
    expect(utils.setEquals(a, b)).toBe(true);

    a = new Set(['a', 'b']);
    b = new Set(['a', 'b']);
    expect(utils.setEquals(a, b)).toBe(true);

    a = new Set(['a', 'b', 'd']);
    b = new Set(['a', 'b', 'd']);
    expect(utils.setEquals(a, b)).toBe(true);
  });

  it('correctly determines that sets are not equal', async () => {
    let a = new Set(['a']);
    let b = new Set(['b']);
    expect(utils.setEquals(a, b)).toBe(false);

    a = new Set(['a', 'd']);
    b = new Set(['a', 'c']);
    expect(utils.setEquals(a, b)).toBe(false);

    a = new Set(['a', 'b', 'd']);
    b = new Set(['a', 'c']);
    expect(utils.setEquals(a, b)).toBe(false);
  });
});

describe('arrayContainsSame test', () => {
  it('correctly determines that empty arrays contain the same items',
     async () => {
       const emptyA: string[] = [];
       const emptyB: string[] = [];
       expect(utils.arrayContainsSame(emptyA, emptyB)).toBe(true);
     });

  it('correctly determines that arrays contain the same items', async () => {
    let a = ['a'];
    let b = ['a'];
    expect(utils.arrayContainsSame(a, b)).toBe(true);

    a = ['a', 'b', 'd'];
    b = ['a', 'b', 'd'];
    expect(utils.arrayContainsSame(a, b)).toBe(true);
  });

  it('works for arrays with different number of duplicates', async () => {
    const a = ['a', 'b'];
    const b = ['a', 'b', 'b', 'b', 'a'];
    expect(utils.arrayContainsSame(a, b)).toBe(true);
  });

  it('correctly determines that arrays do not contain the same items',
     async () => {
       let a = ['a'];
       let b = ['b'];
       expect(utils.arrayContainsSame(a, b)).toBe(false);

       a = ['a', 'b', 'd'];
       b = ['a', 'c', 'd'];
       expect(utils.arrayContainsSame(a, b)).toBe(false);
     });
});

describe('isLitSubtype test', () => {
  it('finds a subclass', () => {
    const spec: Spec = {
      'score': {
        __class__: 'LitType',
        __name__: 'RegressionScore',
        __mro__: ['RegressionScore', 'Scalar', 'LitType', 'object']
      },
    };

    expect(utils.isLitSubtype(spec['score'], 'RegressionScore')).toBe(true);
    expect(utils.isLitSubtype(spec['score'], 'Scalar')).toBe(true);
    expect(utils.isLitSubtype(spec['score'], 'LitType')).toBe(true);
    expect(utils.isLitSubtype(spec['score'], 'TextSegment')).toBe(false);
  });
});

describe('findSpecKeys test', () => {
  const spec: Spec = {
    'score': {
      __class__: 'LitType',
      __name__: 'RegressionScore',
      __mro__: ['RegressionScore', 'Scalar', 'LitType', 'object']
    },
    'probabilities': {
      __class__: 'LitType',
      __name__: 'MulticlassPreds',
      __mro__: ['MulticlassPreds', 'LitType', 'object'],
      null_idx: 0
    },
    'score2': {
      __class__: 'LitType',
      __name__: 'RegressionScore',
      __mro__: ['RegressionScore', 'Scalar', 'LitType', 'object']
    },
    'scalar_foo': {
      __class__: 'LitType',
      __name__: 'Scalar',
      __mro__: ['Scalar', 'LitType', 'object']
    },
    'segment': {
      __class__: 'LitType',
      __name__: 'TextSegment',
      __mro__: ['TextSegment', 'LitType', 'object']
    },
    'generated_text': {
      __class__: 'LitType',
      __name__: 'GeneratedText',
      __mro__: ['GeneratedText', 'TextSegment', 'LitType', 'object'],
      parent: 'segment'
    }
  };

  it('finds all spec keys that match the specified types', () => {
    // Key is in spec.
    expect(utils.findSpecKeys(spec, 'RegressionScore')).toEqual([
      'score', 'score2'
    ]);
    expect(utils.findSpecKeys(spec, 'MulticlassPreds')).toEqual([
      'probabilities'
    ]);

    // Keys are in spec.
    expect(utils.findSpecKeys(spec, [
      'MulticlassPreds', 'RegressionScore'
    ])).toEqual(['score', 'probabilities', 'score2']);
    expect(utils.findSpecKeys(spec, ['GeneratedText'])).toEqual([
      'generated_text'
    ]);

    // Key is not in spec.
    expect(utils.findSpecKeys(spec, ['TokenGradients'])).toEqual([]);
  });

  it('identifies subclass fields', () => {
    expect(utils.findSpecKeys(spec, 'LitType')).toEqual(Object.keys(spec));
    expect(utils.findSpecKeys(spec, 'TextSegment')).toEqual([
      'segment', 'generated_text'
    ]);
    expect(utils.findSpecKeys(spec, 'Scalar')).toEqual([
      'score', 'score2', 'scalar_foo'
    ]);
  });
});


describe('flatten test', () => {
  it('flattens a nested array by a single level', async () => {
    // Empty array
    expect(utils.flatten([])).toEqual([]);

    // Nested empty arrays.
    expect(utils.flatten([[], []])).toEqual([]);

    // Nested arrays.
    expect(utils.flatten([[1, 2], [3]])).toEqual([1, 2, 3]);
    expect(utils.flatten([[1, 2], [], [3], [7, 8, 1]])).toEqual([
      1, 2, 3, 7, 8, 1
    ]);
  });
});


describe('permute test', () => {
  it('permutes an array correctly', async () => {
    expect(utils.permute([], [])).toEqual([]);
    expect(utils.permute([0, 1, 2], [0, 1, 2])).toEqual([0, 1, 2]);
    expect(utils.permute([0, 1, 2], [2, 1, 0])).toEqual([2, 1, 0]);
    expect(utils.permute([5, 6, 7, 8], [3, 1, 0, 2])).toEqual([8, 6, 5, 7]);
  });
});


describe('handleEnterKey test', () => {
  it('Handles input correctly', () => {
    const callback = jasmine.createSpy('callback');
    const event = new KeyboardEvent('keyup', {key: 'Enter'});
    utils.handleEnterKey(event, callback);
    expect(callback).toHaveBeenCalled();
  });

  it('Is  not called for other keys', () => {
    const callback = jasmine.createSpy('callback');
    const event = new KeyboardEvent('keyup', {key: 'Delete'});
    utils.handleEnterKey(event, callback);
    expect(callback).not.toHaveBeenCalled();
  });
});

describe('getThresholdFromMargin test', () => {
  it('Works as expected in the basic case', () => {
    expect(utils.getThresholdFromMargin(-5)).toEqual(0.0066928509242848554);
    expect(utils.getThresholdFromMargin(-2.5)).toEqual(0.07585818002124355);
    expect(utils.getThresholdFromMargin(1)).toEqual(0.7310585786300049);
    expect(utils.getThresholdFromMargin(5 / 3)).toEqual(0.8411308951190849);
    expect(utils.getThresholdFromMargin(5)).toEqual(0.9933071490757153);
  });
  it('Behaves correctly for 0 input', () => {
    expect(utils.getThresholdFromMargin(0)).toEqual(.5);
  });
});

describe('shortenID test', () => {
  it('Shortens an id to 6 characters, for display', () => {
    expect(utils.shortenId('b6ea684bec7bb1d4b9f2736b749c3030'))
        .toEqual('b6ea68');
    expect(utils.shortenId('924d4976b4ac56d053ed956671652892'))
        .toEqual('924d49');
  });
  it('Behaves correctly for null input', () => {
    expect(utils.shortenId(null)).toEqual(undefined);
  });
});

describe('isNumber test', () => {
  it('Returns true for normal numbers', () => {
    expect(utils.isNumber(4)).toEqual(true);
    expect(utils.isNumber(9999999)).toEqual(true);
    expect(utils.isNumber(-9999999)).toEqual(true);
    expect(utils.isNumber(0)).toEqual(true);
  });

  it('Works for text numbers', () => {
    expect(utils.isNumber('0')).toEqual(true);
    expect(utils.isNumber('12')).toEqual(true);
  });

  it('Returns false for infinite numbers', () => {
    expect(utils.isNumber(Number.POSITIVE_INFINITY)).toEqual(false);
    expect(utils.isNumber(Number.NEGATIVE_INFINITY)).toEqual(false);
  });

  it('Returns false for non-numbers', () => {
    expect(utils.isNumber(NaN)).toEqual(false);
    expect(utils.isNumber('asdf')).toEqual(false);
    expect(utils.isNumber('twelve')).toEqual(false);
  });
});

describe('sumArray test', () => {
  it('Correctly sums a normal array', () => {
    let arr = [2, 0, 1];
    expect(utils.sumArray(arr)).toEqual(3);

    arr = [5, 2, 1, 3, 0, 2, 1];
    expect(utils.sumArray(arr)).toEqual(14);
  });
  it('Correctly sums an array with negative numbers', () => {
    const arr = [-1, 3, 0, -4];
    expect(utils.sumArray(arr)).toEqual(-2);
  });
});

describe('range test', () => {
  it('Creates a range() array from a number', () => {
    // Empty array
    expect(utils.range(0)).toEqual([]);

    // Standard arrays
    expect(utils.range(1)).toEqual([0]);
    expect(utils.range(4)).toEqual([0, 1, 2, 3]);
  });
});

describe('cumSumArray test', () => {
  it('cumulatively sums an array', () => {
    // Standard arrays
    let arr = [2, 0, 1];
    expect(utils.cumSumArray(arr)).toEqual([2, 2, 3]);

    arr = [5, 2, 1, 3, 0, 2, 1];
    expect(utils.cumSumArray(arr)).toEqual([5, 7, 8, 11, 11, 13, 14]);
  });
  it('cumulatively sums an array with negative numbers', () => {
    const arr = [-1, 3, 0, -4];
    expect(utils.cumSumArray(arr)).toEqual([-1, 2, 2, -2]);
  });
});

describe('compareArrays test', () => {
  it('Correctly tests normal comparison', () => {
    // Shorter arrays.
    let a = [2, 0];
    let b = [1];
    expect(utils.compareArrays(a, b)).toBe(1);

    // Longer arrays.
    a = [5, 9, 24, 1, 0, 0];
    b = [2, 10, 40, 2, 2];
    expect(utils.compareArrays(a, b)).toBe(1);

    // When a < b.
    a = [2, 10, 40, 2, 2];
    b = [5, 9, 24, 1, 0, 0];
    expect(utils.compareArrays(a, b)).toBe(-1);
  });

  it('Works correctly when b is a prefix of a', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [1, 2, 3, 4];
    expect(utils.compareArrays(a, b)).toBe(1);
  });

  it('Works correctly when arrays are equal', () => {
    // Non-empty arrays.
    let a = [3, 5, 8];
    let b = [3, 5, 8];
    expect(utils.compareArrays(a, b)).toBe(0);

    // Empty arrays.
    a = [];
    b = [];
    expect(utils.compareArrays(a, b)).toBe(0);
  });
});

describe('roundToDecimalPlaces test', () => {
  it('rounds to zero places correctly', () => {
    expect(utils.roundToDecimalPlaces(4.22, 0)).toEqual(4);
    expect(utils.roundToDecimalPlaces(-5.6, 0)).toEqual(-6);
  });

  it('rounds to one place correctly', () => {
    expect(utils.roundToDecimalPlaces(4.54, 1)).toEqual(4.5);
    expect(utils.roundToDecimalPlaces(4.55, 1)).toEqual(4.6);
  });

  it('does not add unnecessary decimals', () => {
    expect(utils.roundToDecimalPlaces(33, 1)).toEqual(33);
  });

  it('rounds to two places correctly', () => {
    expect(utils.roundToDecimalPlaces(4.546, 2)).toEqual(4.55);
    expect(utils.roundToDecimalPlaces(3.333, 2)).toEqual(3.33);
  });

  it('does not round when given negative places', () => {
    expect(utils.roundToDecimalPlaces(4.22, -1)).toEqual(4.22);
  });
});

describe('findMatchingIndices test', () => {
  it('returns empty list when appropriate', () => {
    expect(utils.findMatchingIndices([1, 2, 3], 4)).toEqual([]);
    expect(utils.findMatchingIndices(['a', 'b', 'c'], 'aa')).toEqual([]);
  });

  it('returns one match correctly', () => {
    expect(utils.findMatchingIndices([1, 2, 3], 1)).toEqual([0]);
    expect(utils.findMatchingIndices(['a', 'b', 'c'], 'b')).toEqual([1]);
  });

  it('returns multiple matches correctly', () => {
    expect(utils.findMatchingIndices([1, 2, 3, 1, 1], 1)).toEqual([0, 3, 4]);
  });
});

describe('replaceNth test', () => {
  it('returns same string when no match', () => {
    expect(utils.replaceNth('hello world', 'no', 'yes', 1)).toEqual(
        'hello world');
    expect(utils.replaceNth('hello world', 'world', 'yes', 2)).toEqual(
        'hello world');
    expect(utils.replaceNth('hello world', 'world', 'yes', 0)).toEqual(
        'hello world');
  });

  it('returns correct string on single match', () => {
    expect(utils.replaceNth('hello world', 'world', 'yes', 1)).toEqual(
        'hello yes');
  });

  it('can handle strings with special regex chars', () => {
    expect(utils.replaceNth('hello [MASK]', '[MASK]', 'yes', 1)).toEqual(
        'hello yes');
  });

  it('returns correct string with multiple matches', () => {
    expect(utils.replaceNth('hello world world', 'world', 'yes', 1)).toEqual(
        'hello yes world');
    expect(utils.replaceNth('hello world world', 'world', 'yes', 2)).toEqual(
        'hello world yes');
  });
});
