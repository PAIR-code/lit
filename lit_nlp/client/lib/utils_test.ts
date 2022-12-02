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

import {CategoryLabel, GeneratedText, LitType, MulticlassPreds, RegressionScore, Scalar, StringLitType, TextSegment, TokenGradients} from '../lib/lit_types';
import {mockMetadata, mockSerializedMetadata} from './testing_utils';
import {Spec} from '../lib/types';

import * as utils from './utils';

describe('mean test', () => {
  it('computes a mean', () => {
    const values = [1, 3, 2, 5, 4];
    expect(utils.mean(values)).toEqual(3);
  });
});

describe('median test', () => {
  it('computes a median for a list of integers with odd length', () => {
    const values = [1, 3, 2, 5, 4];
    expect(utils.median(values)).toEqual(3);
  });

  it('computes a median for a list of integers with even length', () => {
    const values = [1, 3, 2, 5, 4, 6];
    expect(utils.median(values)).toEqual(3.5);
  });
});

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

describe('convert lit types to names tests', () => {
  it('converts types to names', () => {
    const types = [Scalar, StringLitType];
    const names = ['Scalar', 'StringLitType'];

    const result = utils.getTypeNames(types);
    expect(result).toEqual(names);
  });

  it('converts names to types', () => {
    const testTypes = [Scalar, StringLitType];
    const names = ['Scalar', 'StringLitType'];

    const result = utils.getTypes(names);
    expect(result).toEqual(testTypes);
    expect(utils.createLitType(StringLitType) instanceof result[1])
        .toBe(true);

  });
});

describe('createLitType test', () => {
  it('creates a type as expected', () => {
    const expected = new Scalar();
    expected.show_in_data_table = false;

    const result = utils.createLitType(Scalar);
    expect(result.name).toEqual('Scalar');
    expect(result).toEqual(expected);
    expect(result instanceof Scalar).toEqual(true);
  });

  it('creates with constructor params', () => {
    const expected = new StringLitType();
    expected.default = 'foo';
    expected.show_in_data_table = true;

    const result = utils.createLitType(
        StringLitType, {'show_in_data_table': true, 'default': 'foo'});
    expect(result).toEqual(expected);
  });

  it('creates a modifiable lit type', () => {
    const result = utils.createLitType(Scalar);
    result.min_val = 5;
    expect(result.min_val).toEqual(5);
  });

  it('handles invalid constructor params', () => {
    expect(() => utils.createLitType(StringLitType, {
      'notAStringParam': true
    })).toThrowError();
  });

  it('creates with constructor params and custom properties', () => {
    const vocab = ['vocab1', 'vocab2'];
    const categoryLabel =
        utils.createLitType(CategoryLabel, {'vocab': vocab});
    expect(categoryLabel.vocab).toEqual(vocab);
  });

  it('allows modification of custom properties', () => {
    const vocab = ['vocab1', 'vocab2'];
    const categoryLabel = utils.createLitType(CategoryLabel);
    categoryLabel.vocab = vocab;
    expect(categoryLabel.vocab).toEqual(vocab);
  });
});

describe('deserializeLitTypesInSpec test', () => {
  const testSpec = {
    'probabilities': {
      '__name__': 'MulticlassPreds',
      'required': true,
      'vocab': ['0', '1'],
      'null_idx': 0,
      'parent': 'label'
    },
    'pooled_embs': {
      '__name__': 'Embeddings',
      'required': true
    }
  };

  it('returns serialized littypes', () => {
    expect(testSpec['probabilities'] instanceof MulticlassPreds)
        .toBe(false);
    const result = utils.deserializeLitTypesInSpec(testSpec);
    expect(result['probabilities'])
        .toEqual(utils.createLitType(
            MulticlassPreds,
            {'vocab': ['0', '1'], 'null_idx': 0, 'parent': 'label'}));
    expect(result['probabilities'] instanceof MulticlassPreds)
        .toBe(true);
  });
});

describe('deserializeLitTypesInLitMetadata test', () => {
  it('deserializes lit metadata', () => {
    const result =
        utils.deserializeLitTypesInLitMetadata(mockSerializedMetadata);
    expect(result).toEqual(mockMetadata);
  });
});

describe('cloneSpec test', () => {
  const testSpec = {
    'probabilities': utils.createLitType(MulticlassPreds, {
      'required': true,
      'vocab': ['0', '1'],
      'null_idx': 0,
      'parent': 'label'
    })
  };

  it('deeply copies', () => {
    const testSpec2 = utils.cloneSpec(testSpec);

    const oldProbs = testSpec['probabilities'];
    const newProbs = testSpec2['probabilities'] as MulticlassPreds;

    newProbs.null_idx = 1;
    expect(oldProbs.null_idx).toEqual(0);
    expect(newProbs.null_idx).toEqual(1);

    expect(testSpec2['probabilities'] instanceof MulticlassPreds)
        .toBe(true);
  });
});

describe('isLitSubtype test', () => {
  it('checks is lit subtype', () => {
    const testType = utils.createLitType(StringLitType);
    expect(utils.isLitSubtype(testType, StringLitType)).toBe(true);
    expect(utils.isLitSubtype(testType, [Scalar])).toBe(false);
  });

  it('finds a subclass', () => {
    const spec: Spec = {
      'score': utils.createLitType(RegressionScore),
    };

    expect(utils.isLitSubtype(spec['score'], [RegressionScore])).toBe(true);
    expect(utils.isLitSubtype(spec['score'], [Scalar])).toBe(true);
    expect(utils.isLitSubtype(spec['score'], [LitType])).toBe(true);
    expect(utils.isLitSubtype(spec['score'], [TextSegment])).toBe(false);
  });
});

describe('findSpecKeys test', () => {
  const spec: Spec = {
    'score': utils.createLitType(RegressionScore),
    'probabilities': utils.createLitType(MulticlassPreds, {'null_idx': 0}),
    'score2': utils.createLitType(RegressionScore),
    'scalar_foo': utils.createLitType(Scalar),
    'segment': utils.createLitType(TextSegment),
    'generated_text':
        utils.createLitType(GeneratedText, {'parent': 'segment'}),
  };

  it('finds all spec keys that match the specified types', () => {
    // Key is in spec.
    expect(utils.findSpecKeys(spec, RegressionScore)).toEqual([
      'score', 'score2'
    ]);
    expect(utils.findSpecKeys(spec, [MulticlassPreds])).toEqual([
      'probabilities'
    ]);

    // Keys are in spec.
    expect(utils.findSpecKeys(spec, [
      MulticlassPreds, RegressionScore
    ])).toEqual(['score', 'probabilities', 'score2']);
    expect(utils.findSpecKeys(spec, [GeneratedText])).toEqual([
      'generated_text'
    ]);

    // Key is not in spec.
    expect(utils.findSpecKeys(spec, [TokenGradients])).toEqual([]);
  });

  it('identifies subclass fields', () => {
    expect(utils.findSpecKeys(spec, LitType)).toEqual(Object.keys(spec));
    expect(utils.findSpecKeys(spec, [TextSegment])).toEqual([
      'segment', 'generated_text'
    ]);
  });

  it('handles empty spec keys', () => {
    expect(utils.findSpecKeys(spec, [])).toEqual([]);
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
    expect(utils.replaceNth('hello world', 'no', 'yes', 1))
        .toEqual('hello world');
    expect(utils.replaceNth('hello world', 'world', 'yes', 2))
        .toEqual('hello world');
    expect(utils.replaceNth('hello world', 'world', 'yes', 0))
        .toEqual('hello world');
  });

  it('returns correct string on single match', () => {
    expect(utils.replaceNth('hello world', 'world', 'yes', 1))
        .toEqual('hello yes');
  });

  it('can handle strings with special regex chars', () => {
    expect(utils.replaceNth('hello [MASK]', '[MASK]', 'yes', 1))
        .toEqual('hello yes');
  });

  it('returns correct string with multiple matches', () => {
    expect(utils.replaceNth('hello world world', 'world', 'yes', 1))
        .toEqual('hello yes world');
    expect(utils.replaceNth('hello world world', 'world', 'yes', 2))
        .toEqual('hello world yes');
  });
});

describe('getStepSizeGivenRange test', () => {
  it('Works as expected in the basic case', () => {
    expect(utils.getStepSizeGivenRange(0.9)).toEqual(0.01);
    expect(utils.getStepSizeGivenRange(1)).toEqual(0.1);
    expect(utils.getStepSizeGivenRange(9.9)).toEqual(0.1);
    expect(utils.getStepSizeGivenRange(10)).toEqual(1);
    expect(utils.getStepSizeGivenRange(100)).toEqual(10);
    expect(utils.getStepSizeGivenRange(101)).toEqual(10);
  });
  it('Behaves correctly for 0 range', () => {
    expect(utils.getStepSizeGivenRange(0)).toEqual(0);
  });
  it('Behaves correctly for negative range', () => {
    expect(utils.getStepSizeGivenRange(-1)).toBeNaN();
  });
});

describe('mapsContainSame test', () => {
  it('correctly determines that empty maps contain the same items', () => {
    const emptyA = new Map<string, string>();
    const emptyB = new Map<string, string>();
    expect(utils.mapsContainSame(emptyA, emptyB)).toBe(true);
  });

  it('correctly determines maps with different values are different', () => {
    const mapA = new Map<string, string>([['keyA', 'valA'], ['keyB', 'valB']]);
    const mapB =
        new Map<string, string>([['keyA', 'valA'], ['keyB', 'valBAlt']]);
    expect(utils.mapsContainSame(mapA, mapB)).toBe(false);
  });

  it('correctly determines maps with different keys are different', () => {
    const mapA = new Map<string, string>([['keyA', 'valA'], ['keyB', 'valB']]);
    const mapB =
        new Map<string, string>([['keyA', 'valA'], ['keyBAlt', 'valB']]);
    expect(utils.mapsContainSame(mapA, mapB)).toBe(false);
  });

  it('correctly determines identical maps are the same', () => {
    const mapA = new Map<string, string>([['keyA', 'valA'], ['keyB', 'valB']]);
    const mapB = new Map<string, string>([['keyA', 'valA'], ['keyB', 'valB']]);
    expect(utils.mapsContainSame(mapA, mapB)).toBe(true);
  });
});

describe('numberRangeFnFromString test', () => {
  it('handles single items', () => {
    const input = '1, 2 -3';
    const fn = utils.numberRangeFnFromString(input);
    expect(fn(1)).toBe(true);
    expect(fn(2)).toBe(true);
    expect(fn(-3)).toBe(true);
    expect(fn(0)).toBe(false);
    expect(fn(-1)).toBe(false);
    expect(fn(3)).toBe(false);
  });

  it('handles ranges', () => {
    const input = '-20--10, -4-4 10-20';
    const fn = utils.numberRangeFnFromString(input);
    expect(fn(-20)).toBe(true);
    expect(fn(-19)).toBe(true);
    expect(fn(-10)).toBe(true);
    expect(fn(-9)).toBe(false);
    expect(fn(0)).toBe(true);
    expect(fn(4)).toBe(true);
    expect(fn(9)).toBe(false);
    expect(fn(12)).toBe(true);
  });

  it('handles mix of singles and ranges', () => {
    const input = '1, 3-5';
    const fn = utils.numberRangeFnFromString(input);
    expect(fn(1)).toBe(true);
    expect(fn(2)).toBe(false);
    expect(fn(3)).toBe(true);
    expect(fn(4)).toBe(true);
    expect(fn(5)).toBe(true);
    expect(fn(6)).toBe(false);
  });

  it('ignores invalid settings', () => {
    const input = 'not a number';
    const fn = utils.numberRangeFnFromString(input);
    expect(fn(-20)).toBe(true);
    expect(fn(0)).toBe(true);
    expect(fn(100)).toBe(true);
  });

  it('handles decimal numbers with and without leading zeros', () => {
    const input = '-11.23--5.5, 0.45-.49, .99-1.4';
    const fn = utils.numberRangeFnFromString(input);
    expect(fn(-11.4)).toBe(false);
    expect(fn(-6.5)).toBe(true);
    expect(fn(0)).toBe(false);
    expect(fn(0.451)).toBe(true);
    expect(fn(0.5)).toBe(false);
    expect(fn(1)).toBe(true);
    expect(fn(1.4)).toBe(true);
    expect(fn(1.5)).toBe(false);
  });
});

describe('linearSpace test', () => {
  it('returns evenly spaced numbers between minValue and maxValue', () => {
    let minValue = 0;
    let maxValue = 1;
    let numSteps = 5;
    let result = [0, 0.25, 0.5, 0.75, 1];
    expect(utils.linearSpace(minValue, maxValue, numSteps)).toEqual(result);

    minValue = -1;
    maxValue = 1;
    numSteps = 5;
    result = [-1, -0.5, 0, 0.5, 1];
    expect(utils.linearSpace(minValue, maxValue, numSteps)).toEqual(result);

    minValue = 0.25;
    maxValue = 0.75;
    numSteps = 3;
    result = [0.25, 0.5, 0.75];
    expect(utils.linearSpace(minValue, maxValue, numSteps)).toEqual(result);

    minValue = 1;
    maxValue = 0.75;
    numSteps = 3;
    result = [];
    expect(utils.linearSpace(minValue, maxValue, numSteps)).toEqual(result);
  });
});
