import 'jasmine';

import * as litTypes from './lit_types';

describe('creates lit types', () => {
  it('with the correct inheritance', () => {
    const testString = new litTypes.StringLitType();
    testString.default = 'string value';

    expect(testString.default).toBe('string value');
    expect(testString.required).toBe(true);

    expect(testString instanceof litTypes.StringLitType).toBe(true);
    expect(testString instanceof litTypes.LitType).toBe(true);
    expect(testString instanceof litTypes.Scalar).toBe(false);
  });

  it('with the correct properties', () => {
    const token = new litTypes.Tokens();
    expect(token.token_prefix).toBe('##');
  });

  it('with undefined properties', () => {
    const generatedText = new litTypes.GeneratedText();
    expect(generatedText.hasOwnProperty('parent')).toBe(true);
    expect(generatedText.parent).toBe(undefined);
  });
});

describe('handles custom methods', () => {
  it('MulticlassPreds num_labels', () => {
    const preds = new litTypes.MulticlassPreds();
    preds.vocab = ['1', '2', '3'];

    expect(preds.num_labels).toBe(3);
    expect(preds.autosort).toBe(false);
  });
});
