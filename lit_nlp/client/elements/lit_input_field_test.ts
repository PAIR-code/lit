import 'jasmine';
import {html, render} from 'lit';
import {LitCheckbox} from './checkbox';
import {LitInputField} from './lit_input_field';
import {NumericInput} from './numeric_input';
import * as litTypes from '../lib/lit_types';

describe('lit input field test', () => {
  describe('readonly mode tests', () => {
    [
      {
        testcaseName:'false',
        litType: new litTypes.BooleanLitType(),
        value: false,
        expected: ' '
      },
      {
        testcaseName:'true',
        litType: new litTypes.BooleanLitType(),
        value: true,
        expected: '✔'
      },
      {
        testcaseName:'EdgeLabels',
        litType: new litTypes.EdgeLabels(),
        value: [{
          span1: [0, 10],
          label: 'correct'
        }],
        expected: '[0,\u00a010)\u2060:\u00a0correct'
      },
      {
        testcaseName:'Embeddings',
        litType: new litTypes.Embeddings(),
        value: [1, 2, 3, 4],
        expected: '<float>[4]'
      },
      {
        testcaseName:'GeneratedTextCandidates with numbers',
        litType: new litTypes.GeneratedTextCandidates(),
        value: [
          [['test1', 0.123], ['test2', 0.234]],
          [['test3', 0.345], ['test4', 0.456]]
        ],
        expected: 'test1 (0.123)\n\ntest2 (0.234)\n\n' +
                  'test3 (0.345)\n\ntest4 (0.456)'
      },
      {
        testcaseName:'GeneratedTextCandidates without numbers',
        litType: new litTypes.GeneratedTextCandidates(),
        value: [
          [['test1', null], ['test2', null]],
          [['test3', null], ['test4', null]]
        ],
        expected: 'test1\n\ntest2\n\ntest3\n\ntest4'
      },
      {
        testcaseName:'MultiSegmentAnnotations with scores',
        litType: new litTypes.MultiSegmentAnnotations(),
        value: [
          {label: 'label_1', score: 0.123, spans: []},
          {label: 'label_2', score: 1.234, spans: []},
          {label: 'label_3', score: 2.345, spans: []},
        ],
        expected: 'label_1 (0.123), label_2 (1.234), label_3 (2.345)'
      },
      {
        testcaseName:'MultiSegmentAnnotations without scores',
        litType: new litTypes.MultiSegmentAnnotations(),
        value: [
          {label: 'label_1', spans: []},
          {label: 'label_2', spans: []},
          {label: 'label_3', spans: []},
        ],
        expected: 'label_1, label_2, label_3'
      },
      {
        testcaseName:'Scalar',
        litType: new litTypes.Scalar(),
        value: 0.9876,
        expected: '0.988'
      },
      {
        testcaseName:'Integer',
        litType: new litTypes.Integer(),
        value: 1,
        expected: '1'
      },
      {
        testcaseName:'a number value',
        litType: new litTypes.LitType(),
        value: 0.9876,
        expected: '0.988'
      },
      {
        testcaseName:'StringLitType as a string',
        litType: new litTypes.StringLitType(),
        value: 'a test string',
        expected: 'a test string'
      },
      {
        testcaseName:'a string value as a string',
        litType: new litTypes.LitType(),
        value: 'a test string',
        expected: 'a test string'
      },
      {
        testcaseName:'SpanLabels without scores',
        litType: new litTypes.SpanLabels(),
        value: [
          {start: 0, end: 1},
          {start: 0, end: 1, align: 'field_name'},
          {start: 0, end: 1, label: 'label'},
          {start: 0, end: 1, align: 'field_name', label: 'label'},
        ],
        expected: '[0,\u00a01), ' +
                  'field_name\u00a0[0,\u00a01), ' +
                  '[0,\u00a01)\u2060:\u00a0label, ' +
                  'field_name\u00a0[0,\u00a01)\u2060:\u00a0label',
      },
      {
        testcaseName:'arrays of numbers',
        litType: new litTypes.LitType(),
        value: [1, 2, 3, 4],
        expected: '1, 2, 3, 4'
      },
      {
        testcaseName:'arrays of ScoredTextCandidate with a number',
        litType: new litTypes.LitType(),
        value: ['test', 0.123],
        expected: 'test (0.123)'
      },
      {
        testcaseName:'arrays of ScoredTextCandidate without a number',
        litType: new litTypes.LitType(),
        value: ['test', null],
        expected: 'test'
      },
      {
        testcaseName:'arrays of ScoredTextCandidates with numbers',
        litType: new litTypes.LitType(),
        value: [['test1', 0.123], ['test2', 0.234]],
        expected: 'test1 (0.123)\n\ntest2 (0.234)'
      },
      {
        testcaseName:'arrays of ScoredTextCandidates without numbers',
        litType: new litTypes.LitType(),
        value: [['test1', null], ['test2', null]],
        expected: 'test1\n\ntest2'
      },
      {
        testcaseName:'arrays of strings',
        litType: new litTypes.LitType(),
        value: ['test1', 'test2'],
        expected: 'test1, test2'
      }
    ].forEach(({testcaseName, litType, value, expected}) => {
      it(`renders ${testcaseName} as string-like content`, async () => {
        const content = html`
          <lit-input-field
            readonly
            name=${testcaseName}
            .type=${litType}
            .value=${value}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = `lit-input-field[name="${testcaseName}"]`;
        const inputField =
            document.body.querySelector<LitInputField>(queryString);
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField!.updateComplete;
        expect(inputField!.renderRoot.textContent).toBe(expected);
      });
    });

    // TODO(b/273479584): The following test cases include non-printing white
    // space characters in their expected values. Preserve these characters in
    // any edits to this test until the chunkWords() function is replaced.
    [
      {
        testcaseName:'long words',
        litType: new litTypes.StringLitType(),
        value: 'Abetalipoproteinemia',
        expected: 'Abetalipoprotei​nemia',
      },
      {
        testcaseName:'arrays of long words',
        litType: new litTypes.StringList(),
        value: [
          'Abetalipoproteinemia',
          'Adrenocorticotrophic'
        ],
        expected: 'Abetalipoprotei​nemia, Adrenocorticotr​ophic'
      },
    ].forEach(({testcaseName, litType, value, expected}) => {
      it(`respects the limitWords attribute for ${testcaseName}`, async () => {
        const content = html`
          <lit-input-field
            readonly
            limitWords
            name=${testcaseName}
            .type=${litType}
            .value=${value}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = `lit-input-field[name="${testcaseName}"]`;
        const inputField =
            document.body.querySelector<LitInputField>(queryString);
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField!.updateComplete;
        expect(inputField!.renderRoot.textContent).toBe(expected);
      });
    });

    it('renders unsupported types as the empty string', async () => {
      const value = {};
      const content = html`
        <lit-input-field
          readonly
          name="unsupported-object"
          .type=${new litTypes.LitType()}
          .value=${value}>
        </lit-input-field>`;
      render(content, document.body);
      const queryString = 'lit-input-field[name="unsupported-object"]';
      const inputField =
          document.body.querySelector<LitInputField>(queryString);
      expect(inputField).toBeInstanceOf(LitInputField);
      await inputField!.updateComplete;
      expect(inputField!.renderRoot.textContent).toBe('');
    });
  });

  describe('editable tests', () => {
    const CHANGE_HANDLERS = {
      boolean: (e: Event) => {},
      categorical: (e: Event) => {},
      scalar: (e: Event) => {},
      string: (e: Event) => {},
      textSegment: (e: Event) => {},
    };

    describe('boolean types test', () => {
      const booleanType = new litTypes.BooleanLitType();

      it('renders a boolean as a LitCheckbox', async () => {
        const content = html`
          <lit-input-field
            name="boolean-editable"
            .type=${booleanType}
            .value=${true}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="boolean-editable"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const checkbox =
            inputField.renderRoot.querySelector<LitCheckbox>('lit-checkbox')!;
        expect(checkbox).toBeInstanceOf(LitCheckbox);
        expect(checkbox.checked).toBeTrue();
      });

      it('reacts to click events on the LitCheckbox', async () => {
        const spy = spyOn(CHANGE_HANDLERS, 'boolean');
        const content = html`
          <lit-input-field
            name="boolean-editable-changes"
            .type=${booleanType}
            .value=${true}
            @change=${CHANGE_HANDLERS.boolean}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="boolean-editable-changes"]';
        const inputField =
          document.body.querySelector<LitInputField>(queryString)!;
        await inputField.updateComplete;
        expect(inputField.value).toBeTrue();
        const checkbox =
          inputField.renderRoot.querySelector<LitCheckbox>('lit-checkbox')!;
        checkbox.dispatchEvent(new Event('change'));
        await inputField.updateComplete;
        expect(spy).toHaveBeenCalled();
      });
    });

    describe('categorical types tests', () => {
      const catLabelType = new litTypes.CategoryLabel();
      catLabelType.vocab = ['0', '1', '2', '3'];

      it('renders a CategoryLabel with vocab as a select', async () => {
        const content = html`
          <lit-input-field
            name="category-label-editable"
            .type=${catLabelType}
            .value=${'0'}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="category-label-editable"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const select =
            inputField.renderRoot.querySelector<HTMLSelectElement>(
              'select.dropdown')!;
        expect(select).toBeInstanceOf(HTMLSelectElement);
        expect(select.value).toBe('0');
        expect(select.selectedIndex).toBe(0);
      });

      it('reacts to change events on the select', async () => {
        const spy = spyOn(CHANGE_HANDLERS, 'categorical');
        const content = html`
          <lit-input-field
            name="category-label-editable"
            .type=${catLabelType}
            .value=${'0'}
            @change=${CHANGE_HANDLERS.categorical}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="category-label-editable"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        await inputField.updateComplete;
        const select =
            inputField.renderRoot.querySelector<HTMLSelectElement>(
              'select.dropdown')!;
        select.dispatchEvent(new Event('change'));
        await inputField.updateComplete;
        expect(spy).toHaveBeenCalled();
      });

      it('renders a CategoryLabel without vocab as an input', async () => {
        const name = 'category-label-editable-no-vocab';
        const content = html`
          <lit-input-field
            name=${name}
            .type=${new litTypes.CategoryLabel()}
            .value=${'0'}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = `lit-input-field[name="${name}"]`;
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const input =
            inputField.renderRoot.querySelector<HTMLInputElement>('input')!;
        expect(input).toBeInstanceOf(HTMLInputElement);
        expect(input.value).toBe('0');
        expect(input.type).toBe('text');
      });

      it('reacts to input events on the input', async () => {
        const name = 'category-label-editable-no-vocab';
        const spy = spyOn(CHANGE_HANDLERS, 'categorical');
        const content = html`
          <lit-input-field
            name=${name}
            .type=${new litTypes.CategoryLabel()}
            .value=${'0'}
            @change=${CHANGE_HANDLERS.categorical}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = `lit-input-field[name="${name}"]`;
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const input =
            inputField.renderRoot.querySelector<HTMLInputElement>('input')!;
        input.dispatchEvent(new Event('input'));
        await inputField.updateComplete;
        expect(spy).toHaveBeenCalled();
      });
    });

    describe('scalar types tests', () => {
      [
        {
          testcaseName: 'float',
          litType: new litTypes.Scalar(),
          value: 0.123,
        },
        {
          testcaseName: 'integer',
          litType: new litTypes.Integer(),
          value: 4,
        },
      ].forEach(({testcaseName, litType, value}) => {
        it(`renders a ${testcaseName} as a LitNumericInput`, async () => {
          const name = `${testcaseName}-editable`;
          const content = html`
            <lit-input-field
              name=${name}
              .type=${litType}
              .value=${value}>
            </lit-input-field>`;
          render(content, document.body);
          const queryString = `lit-input-field[name="${name}"]`;
          const inputField =
              document.body.querySelector<LitInputField>(queryString)!;
          expect(inputField).toBeInstanceOf(LitInputField);
          await inputField.updateComplete;
          const numericInput =
              inputField.renderRoot.querySelector<NumericInput>('lit-numeric-input')!;
          expect(numericInput).toBeInstanceOf(NumericInput);
          expect(numericInput.value).toBe(value);
        });
      });

      it('reacts to change events on the NumericInput', async () => {
        const spy = spyOn(CHANGE_HANDLERS, 'scalar');
        const content = html`
          <lit-input-field
            name="scalar-editable-changes"
            .type=${new litTypes.Scalar()}
            .value=${0}
            @change=${CHANGE_HANDLERS.scalar}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="scalar-editable-changes"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        await inputField.updateComplete;
        const numericInput =
            inputField.renderRoot.querySelector<NumericInput>('lit-numeric-input')!;
        const sliderInput =
            numericInput.renderRoot.querySelector<HTMLInputElement>('input[type="range"].slider')!;
        sliderInput.value = '1';
        sliderInput.dispatchEvent(new Event('change'));
        await numericInput.updateComplete;
        expect(spy).toHaveBeenCalled();
      });
    });

    describe('string types tests', () => {
      it('renders a StringLitType as an input[type="text"]', async () => {
        const content = html`
          <lit-input-field
            name="string-editable"
            .type=${new litTypes.StringLitType()}
            .value=${'test string'}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="string-editable"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const textarea =
            inputField.renderRoot.querySelector<HTMLInputElement>(
              'input.input-short')!;
        expect(textarea).toBeInstanceOf(HTMLInputElement);
        expect(textarea.type).toBe('text');
        expect(textarea.value).toBe('test string');
      });

      it('reacts to input changes on the input[type="text"]', async () => {
        const spy = spyOn(CHANGE_HANDLERS, 'string');
        const content = html`
          <lit-input-field
            name="string-editable-changes"
            .type=${new litTypes.StringLitType()}
            .value=${'test string'}
            @change=${CHANGE_HANDLERS.string}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="string-editable-changes"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        await inputField.updateComplete;
        const textarea =
            inputField.renderRoot.querySelector<HTMLInputElement>(
              'input.input-short')!;
        textarea.dispatchEvent(new Event('input'));
        await inputField.updateComplete;
        expect(spy).toHaveBeenCalled();
      });

      it('renders a TextSegment as a textarea', async () => {
        const content = html`
          <lit-input-field
            name="text-segment-editable"
            .type=${new litTypes.TextSegment()}
            .value=${'test string'}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString = 'lit-input-field[name="text-segment-editable"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        expect(inputField).toBeInstanceOf(LitInputField);
        await inputField.updateComplete;
        const textarea =
            inputField.renderRoot.querySelector<HTMLTextAreaElement>(
              'textarea.input-box')!;
        expect(textarea).toBeInstanceOf(HTMLTextAreaElement);
        expect(textarea.value).toBe('test string');
      });

      it('reacts to input changes on the textarea', async () => {
        const spy = spyOn(CHANGE_HANDLERS, 'textSegment');
        const content = html`
          <lit-input-field
            name="text-segment-editable-changes"
            .type=${new litTypes.TextSegment()}
            .value=${'test string'}
            @change=${CHANGE_HANDLERS.textSegment}>
          </lit-input-field>`;
        render(content, document.body);
        const queryString =
            'lit-input-field[name="text-segment-editable-changes"]';
        const inputField =
            document.body.querySelector<LitInputField>(queryString)!;
        await inputField.updateComplete;
        const textarea =
            inputField.renderRoot.querySelector<HTMLTextAreaElement>(
              'textarea.input-box')!;
        textarea.dispatchEvent(new Event('input'));
        await inputField.updateComplete;
        expect(spy).toHaveBeenCalled();
      });
    });
  });

  describe('labeling tests', () => {
    const litType = new litTypes.StringLitType();
    const value = 'test string';

    it('does not render a label by default', async () => {
      const content = html`
        <lit-input-field
          name="no-label"
          .type=${litType}
          .value=${value}>
        </lit-input-field>`;
      render(content, document.body);
      const queryString = 'lit-input-field[name="no-label"]';
      const inputField =
          document.body.querySelector<LitInputField>(queryString)!;
      expect(inputField).toBeInstanceOf(LitInputField);
      await inputField.updateComplete;
      const [firstChild] = inputField.renderRoot.children;
      expect(firstChild).not.toBeInstanceOf(HTMLLabelElement);
    });

    it('renders a label with the text on the left by default', async () => {
      const content = html`
        <lit-input-field
          name="label-left"
          label="left-side label"
          .type=${litType}
          .value=${value}>
        </lit-input-field>`;
      render(content, document.body);
      const queryString = 'lit-input-field[name="label-left"]';
      const inputField =
          document.body.querySelector<LitInputField>(queryString)!;
      expect(inputField).toBeInstanceOf(LitInputField);
      await inputField.updateComplete;
      const [labelElem] = inputField.renderRoot.children;
      expect(labelElem).toBeInstanceOf(HTMLLabelElement);
      expect(labelElem.textContent).toBe('left-side label');
      console.log(labelElem.children);
    });

    it('can renders a label with the text on the right', async () => {
      const content = html`
        <lit-input-field
          name="label-right"
          label="right-side label"
          labelPlacement="right"
          .type=${litType}
          .value=${value}>
        </lit-input-field>`;
      render(content, document.body);
      const queryString = 'lit-input-field[name="label-right"]';
      const inputField =
          document.body.querySelector<LitInputField>(queryString)!;
      expect(inputField).toBeInstanceOf(LitInputField);
      await inputField.updateComplete;
      const [labelElem] = inputField.renderRoot.children;
      expect(labelElem).toBeInstanceOf(HTMLLabelElement);
      expect(labelElem.textContent).toBe('right-side label');
    });
  });
});
