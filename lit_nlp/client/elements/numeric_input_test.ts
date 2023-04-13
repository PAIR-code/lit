import 'jasmine';

import {LitElement} from 'lit';

import {NumericInput} from './numeric_input';

describe('numeric_input test', () => {
    let numericInput: NumericInput;

    beforeEach(async () => {
      numericInput = new NumericInput();
      document.body.appendChild(numericInput);
      await numericInput.updateComplete;
    });

    it('should instantiate correctly', () => {
      expect(numericInput).toBeDefined();
      expect(numericInput instanceof HTMLElement).toBeTrue();
      expect(numericInput instanceof LitElement).toBeTrue();
    });

    [
      {testcaseName: 1, step: 0.1, value: 0.7, expected: '0.7'},
      {testcaseName: 2, step: 0.01, value: 0.07, expected: '0.07'},
      {testcaseName: 3, step: 0.001, value: 0.007, expected: '0.007'}
    ].forEach(({testcaseName, step, value, expected}) => {
      it(`should use precision=${testcaseName}`, async () => {
        numericInput.value = value;
        numericInput.step = step;
        await numericInput.updateComplete;
        const sliderVal = numericInput.renderRoot.querySelector
          <HTMLInputElement>('input.slider')!;
        const numericVal = numericInput.renderRoot.querySelector
          <HTMLInputElement>('input[type="number"]')!;
        expect(sliderVal.value).toEqual(expected);
        expect(numericVal.value).toEqual(expected);
      });
    });
  });