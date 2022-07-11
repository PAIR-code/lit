import 'jasmine';

import {Spec} from '../lib/types';

import * as litTypes from './lit_types';
import * as litTypesUtils from './lit_types_utils';


describe('createLitType test', () => {
  it('creates a type as expected', () => {
    const expected = new litTypes.Scalar();
    expected.__name__ = 'Scalar';
    expected.__mro__ = ['Scalar', 'LitType', 'Object'];
    expected.show_in_data_table = false;

    const result = litTypesUtils.createLitType('Scalar');
    expect(result).toEqual(expected);
    expect(result instanceof litTypes.Scalar).toEqual(true);
  });

  it('creates with constructor params', () => {
    const expected = new litTypes.String();
    expected.__name__ = 'String';
    expected.__mro__ = ['String', 'LitType', 'Object'];
    expected.default = 'foo';
    expected.show_in_data_table = true;

    const result = litTypesUtils.createLitType(
        'String', {'show_in_data_table': true, 'default': 'foo'});
    expect(result).toEqual(expected);
  });

  it('creates a modifiable lit type', () => {
    const result = litTypesUtils.createLitType('Scalar');
    result.min_val = 5;
    expect(result.min_val).toEqual(5);
  });

  it('handles invalid constructor params', () => {
    expect(() => litTypesUtils.createLitType('String', {
      'notAStringParam': true
    })).toThrowError();
  });

  it('creates with constructor params and custom properties', () => {
    const vocab = ['vocab1', 'vocab2'];
    const categoryLabel =
        litTypesUtils.createLitType('CategoryLabel', {'vocab': vocab});
    expect(categoryLabel.vocab).toEqual(vocab);
  });

  it('allows modification of custom properties', () => {
    const vocab = ['vocab1', 'vocab2'];
    const categoryLabel = litTypesUtils.createLitType('CategoryLabel');
    categoryLabel.vocab = vocab;
    expect(categoryLabel.vocab).toEqual(vocab);
  });

  it('populates mro', () => {
    let testType = new litTypes.String();
    expect(litTypesUtils.getMethodResolutionOrder(testType)).toEqual([
      'String', 'LitType', 'Object'
    ]);

    testType = new litTypes.LitType();
    expect(litTypesUtils.getMethodResolutionOrder(testType)).toEqual([
      'LitType', 'Object'
    ]);
  });

  it('handles invalid names', () => {
    expect(() => litTypesUtils.createLitType('notLitType')).toThrowError();
  });
});

describe('isLitSubtype test', () => {
  it('checks is lit subtype', () => {
    const testType = new litTypes.String();
    expect(litTypesUtils.isLitSubtype(testType, 'String')).toBe(true);
    expect(litTypesUtils.isLitSubtype(testType, ['String'])).toBe(true);
    expect(litTypesUtils.isLitSubtype(testType, ['Scalar'])).toBe(false);

    // LitType is not a subtype of LitType.
    expect(() => litTypesUtils.isLitSubtype(testType, 'LitType'))
        .toThrowError();
    expect(() => litTypesUtils.isLitSubtype(testType, ['NotAType']))
        .toThrowError();
  });
});


describe('findSpecKeys test', () => {
  // TODO(cjqian): Add original utils_test test after adding more types.
  const spec: Spec = {
    'scalar_foo': new litTypes.Scalar(),
    'segment': new litTypes.String(),
    'generated_text': new litTypes.String(),
  };


  it('finds all spec keys that match the specified types', () => {
    // Key is in spec.
    expect(litTypesUtils.findSpecKeys(spec, 'String')).toEqual([
      'segment', 'generated_text'
    ]);

    // Keys are in spec.
    expect(litTypesUtils.findSpecKeys(spec, ['String', 'Scalar'])).toEqual([
      'scalar_foo', 'segment', 'generated_text'
    ]);
  });

  it('handles empty spec keys', () => {
    expect(litTypesUtils.findSpecKeys(spec, [])).toEqual([]);
  });

  it('handles invalid spec keys', () => {
    expect(() => litTypesUtils.findSpecKeys(spec, '')).toThrowError();
    expect(() => litTypesUtils.findSpecKeys(spec, 'NotAType')).toThrowError();
  });
});

describe('tryCastAsType test', () => {
  const litString = litTypesUtils.createLitType('String');
  const textSegment = litTypesUtils.createLitType('TextSegment');
  it('conditionally throws error if fails on an invalid type', () => {
    expect(litTypesUtils.tryCastAsType(3, 'String')).toEqual(null);
    expect(() => litTypesUtils.tryCastAsType(3, 'String', true)).toThrowError();
  });

  it('conditionally throw error if fails on a missing type', () => {
    expect(litTypesUtils.tryCastAsType(litString, ['TextSegment'])).toEqual(null);
    expect(() => litTypesUtils.tryCastAsType(litString, 'TextSegment', true))
        .toThrowError();
  });

  it('tries to cast a single type', () => {
    expect(litTypesUtils.tryCastAsType(textSegment, 'TextSegment'))
        .toEqual(textSegment);
    expect(litTypesUtils.tryCastAsType(textSegment, 'String'))
        .toEqual((textSegment as litTypes.String));
    expect(litTypesUtils.tryCastAsType(textSegment, 'Scalar'))
        .toEqual(null);
  });

  it('returns the first valid type in a list', () => {
    expect(litTypesUtils.tryCastAsType(textSegment, [
      'Scalar', 'TextSegment', 'String'
    ])).toEqual(textSegment);
    expect(litTypesUtils.tryCastAsType(textSegment, [
      'Scalar', 'String', 'TextSegment'
    ])).toEqual(textSegment as litTypes.String);
  });
});
