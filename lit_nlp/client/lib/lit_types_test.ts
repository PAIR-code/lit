import 'jasmine';

import * as litTypes from './lit_types';

describe('creates lit types', () => {
  it('with the correct inheritance', () => {
    const testString = new litTypes.String();
    testString.default = 'string value';

    expect(testString.default).toBe('string value');
    expect(testString.required).toBe(true);

    expect(testString instanceof litTypes.String).toBe(true);
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
