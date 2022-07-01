import 'jasmine';
import * as litTypes from './lit_types';

describe('lit types test', () => {

  it('creates a string', () => {
    const testString = new litTypes.String();
    testString.default = "string value";

    expect(testString.default).toBe("string value");
    expect(testString.required).toBe(true);

    expect(testString instanceof litTypes.String).toBe(true);
    expect(testString instanceof litTypes.LitType).toBe(true);
    expect(testString instanceof litTypes.Scalar).toBe(false);
  });
});
