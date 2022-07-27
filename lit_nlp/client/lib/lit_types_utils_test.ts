import 'jasmine';

import * as litTypes from './lit_types';
import * as litTypesUtils from './lit_types_utils';


describe('createLitType test', () => {
  it('creates a type as expected', () => {
    const expected = new litTypes.Scalar();
    expected.__name__ = 'Scalar';
    expected.show_in_data_table = false;

    const result = litTypesUtils.createLitType('Scalar');
    expect(result).toEqual(expected);
    expect(result instanceof litTypes.Scalar).toEqual(true);
  });

  it('creates with constructor params', () => {
    const expected = new litTypes.StringLitType();
    expected.__name__ = 'StringLitType';
    expected.default = 'foo';
    expected.show_in_data_table = true;

    const result = litTypesUtils.createLitType(
        'StringLitType', {'show_in_data_table': true, 'default': 'foo'});
    expect(result).toEqual(expected);
  });

  it('creates a modifiable lit type', () => {
    const result = litTypesUtils.createLitType('Scalar');
    result.min_val = 5;
    expect(result.min_val).toEqual(5);
  });

  it('handles invalid constructor params', () => {
    expect(() => litTypesUtils.createLitType('StringLitType', {
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

  it('handles invalid names', () => {
    expect(() => litTypesUtils.createLitType('notLitType')).toThrowError();
  });
});

describe('deserializeLitTypesInSpec test', () => {
  // TODO(b/162269499): Add test for deserializeLitTypesInLitMetadata.
  const testSpec = {
    'probabilities': {
      '__class__': 'LitType',
      '__name__': 'MulticlassPreds',
      '__mro__': ['MulticlassPreds', 'LitType', 'object'],
      'required': true,
      'vocab': ['0', '1'],
      'null_idx': 0,
      'parent': 'label'
    },
    'pooled_embs': {
      '__class__': 'LitType',
      '__name__': 'Embeddings',
      '__mro__': ['Embeddings', 'LitType', 'object'],
      'required': true
    }
  };

  it('returns serialized littypes', () => {
    expect(testSpec['probabilities'] instanceof litTypes.MulticlassPreds)
        .toBe(false);
    const result = litTypesUtils.deserializeLitTypesInSpec(testSpec);
    expect(result['probabilities'])
        .toEqual(litTypesUtils.createLitType(
            'MulticlassPreds',
            {'vocab': ['0', '1'], 'null_idx': 0, 'parent': 'label'}));
    expect(result['probabilities'] instanceof litTypes.MulticlassPreds)
        .toBe(true);
  });
});
