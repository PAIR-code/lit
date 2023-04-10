import 'jasmine';

import {render} from 'lit';
import {AnnotationCluster, EdgeLabel, ScoredTextCandidates, SpanLabel} from './dtypes';
import {formatAnnotationCluster, formatAnnotationClusters, formatBoolean, formatEdgeLabel, formatEmbeddings, formatNumber, formatScoredTextCandidate, formatScoredTextCandidates, formatScoredTextCandidatesList, formatSpanLabel, formatSpanLabels, renderCategoricalInput, renderTextInputLong, renderTextInputShort} from './type_rendering';

describe('Tests for readonly data formatting', () => {
  describe('AnnotationCluster formatting test', () => {
    it('formats an AnnotationCluster with a label and score', () => {
      const label = 'some_label';
      const score = 0.123;
      const formatted = formatAnnotationCluster({label, score, spans: []});
      expect(formatted).toBe(`${label} (${score})`);
    });

    it('formats an AnnotationCluster with only a label', () => {
      const label = 'some_label';
      const formatted = formatAnnotationCluster({label, spans: []});
      expect(formatted).toBe(label);
    });

    it('formats a list of AnnotationClusters with labels and scores', () => {
      const clusters: AnnotationCluster[] = [
        {label: 'label_1', score: 0.123, spans: []},
        {label: 'label_2', score: 1.234, spans: []},
        {label: 'label_3', score: 2.345, spans: []},
      ];
      const formatted = formatAnnotationClusters(clusters);
      render(formatted, document.body);
      const multiSegAnnoDiv =
          document.body.querySelector('div.multi-segment-annotation');

      expect(multiSegAnnoDiv).toBeInstanceOf(HTMLDivElement);
      Array.from(multiSegAnnoDiv!.children).forEach((child, i) => {
        expect(child).toBeInstanceOf(HTMLDivElement);
        expect(child).toHaveClass('annotation-cluster');
        const [div, list] = child.children;
        const {label, score} = clusters[i];
        expect(div).toBeInstanceOf(HTMLDivElement);
        expect(div.textContent).toBe(`${label} (${score})`);
        expect(list).toBeInstanceOf(HTMLUListElement);
      });
    });

    it('formats a list of AnnotationClusters with only labels', () => {
      const clusters: AnnotationCluster[] = [
        {label: 'label_1', spans: []},
        {label: 'label_2', spans: []},
        {label: 'label_3', spans: []},
      ];
      const formatted = formatAnnotationClusters(clusters);
      render(formatted, document.body);
      const multiSegAnnoDiv =
          document.body.querySelector('div.multi-segment-annotation');

      expect(multiSegAnnoDiv).toBeInstanceOf(HTMLDivElement);
      Array.from(multiSegAnnoDiv!.children).forEach((child, i) => {
        expect(child).toBeInstanceOf(HTMLDivElement);
        expect(child).toHaveClass('annotation-cluster');
        const [div, list] = child.children;
        expect(div).toBeInstanceOf(HTMLDivElement);
        expect(div.textContent).toBe(clusters[i].label);
        expect(list).toBeInstanceOf(HTMLUListElement);
      });
    });

  });

  describe('Boolean formatting test', () => {
    [
      {value: false, expected: ' '},
      {value: true, expected: '✔'}
    ].forEach(({value, expected}) => {
      it(`should format '${value}' as '${expected}'`, () => {
        expect(formatBoolean(value)).toBe(expected);
      });
    });
  });

  describe('EdgeLabel formatting test', () => {
    it('formats an EdgeLabel with 1 span', () => {
      const label: EdgeLabel = {
        span1: [0, 10],
        label: 'correct'
      };
      const formatted = formatEdgeLabel(label);
      const expected = `[0,\u00a010)\u2060:\u00a0correct`;
      expect(formatted).toBe(expected);
    });

    it('formats an EdgeLabel with 2 spans', () => {
      const label: EdgeLabel = {
        span1: [0, 10],
        span2: [10, 20],
        label: 'correct'
      };
      const formatted = formatEdgeLabel(label);
      const expected =
          `[0,\u00a010)\u00a0←\u00a0[10,\u00a020)\u2060:\u00a0correct`;
      expect(formatted).toBe(expected);
    });
  });

  describe('Embeddings formatting test', () => {
    [
      {value: undefined, expected: ''},
      {value: null, expected: ''},
      {value: [], expected: '<float>[0]'},
      {value: [1], expected: '<float>[1]'},
      {value: [1, 2, 3, 4], expected: '<float>[4]'},
    ].forEach(({value, expected}) => {
      const name = value != null ? `[${value}]` : value;
      it(`should format ${name} as '${expected}'`, () => {
        expect(formatEmbeddings(value)).toBe(expected);
      });
    });
  });

  describe('Number formatting test', () => {
    [
      {value: 1, expected: 1},
      {value: 1.0, expected: 1},
      {value: 1.001, expected: 1.001},
      {value: 1.0001, expected: 1},
      {value: 1.0009, expected: 1.001},
    ].forEach(({value, expected}) => {
      it(`should format '${value}' as '${expected}'`, () => {
        expect(formatNumber(value)).toBe(expected);
      });
    });
  });

  describe('ScoredTextCandidate formatting test', () => {
    it('formats a ScoredTextCandidate with text', () => {
      const text = 'some test text';
      expect(formatScoredTextCandidate([text, null])).toBe(text);
    });

    it('formats a ScoredTextCandidate with text and score', () => {
      const text = 'some test text';
      const score = 0.123;
      expect(formatScoredTextCandidate([text, score])).toBe(`${text} (${score})`);
    });

    it('formats ScoredTextCandidates with text', () => {
      const candidates: ScoredTextCandidates = [
        ['text_1', null],
        ['text_2', null],
        ['text_3', null],
      ];
      const expected = 'text_1\n\ntext_2\n\ntext_3';
      expect(formatScoredTextCandidates(candidates)).toBe(expected);
    });

    it('formats ScoredTextCandidates with text and scores', () => {
      const candidates: ScoredTextCandidates = [
        ['text_1', 0.123],
        ['text_2', 1.234],
        ['text_3', 2.345],
      ];
      const expected = 'text_1 (0.123)\n\ntext_2 (1.234)\n\ntext_3 (2.345)';
      expect(formatScoredTextCandidates(candidates)).toBe(expected);
    });

    it('formats a ScoredTextCandidates[] with text', () => {
      const candidates: ScoredTextCandidates[] = [
        [['text_1', null], ['text_2', null], ['text_3', null]],
        [['text_1', null], ['text_2', null], ['text_3', null]],
      ];
      const expected = 'text_1\n\ntext_2\n\ntext_3';
      const formatted = formatScoredTextCandidatesList(candidates);
      expect(formatted).toBe(`${expected}\n\n${expected}`);
    });

    it('formats a ScoredTextCandidates[] with text and scores', () => {
      const candidates: ScoredTextCandidates[] = [
        [['text_1', 0.123], ['text_2', 1.234], ['text_3', 2.345]],
        [['text_1', 0.123], ['text_2', 1.234], ['text_3', 2.345]],
      ];
      const expected = 'text_1 (0.123)\n\ntext_2 (1.234)\n\ntext_3 (2.345)';
      const formatted = formatScoredTextCandidatesList(candidates);
      expect(formatted).toBe(`${expected}\n\n${expected}`);
    });
  });

  describe('SpanLabel formatting test', () => {
    const labels: SpanLabel[] = [
      {start: 0, end: 1},
      {start: 0, end: 1, align: 'field_name'},
      {start: 0, end: 1, label: 'label'},
      {start: 0, end: 1, align: 'field_name', label: 'label'},
    ];

    const expected: string[] = [
      '[0,\u00a01)',
      'field_name\u00a0[0,\u00a01)',
      '[0,\u00a01)\u2060:\u00a0label',
      'field_name\u00a0[0,\u00a01)\u2060:\u00a0label',
    ];

    labels.forEach((label, i) => {
      it('converts a SpanLabel as a string by default', () => {
        const formatted = formatSpanLabel(label);
        expect(formatted).toBe(expected[i]);
      });

      it('converts a SpanLabel to a <div> when monospace=true', () => {
        const formatted = formatSpanLabel(label, true);
        render(formatted, document.body);
        const div = document.body.querySelector('div.monospace-label');
        expect(div).toBeInstanceOf(HTMLDivElement);
        expect(div!.textContent).toBe(expected[i]);
      });
    });

    it('converts a SpanLabel[] to a <div> of <div>s', () => {
      const formatted = formatSpanLabels(labels);
      render(formatted, document.body);
      const div = document.body.querySelector('div.span-labels');

      expect(div).toBeInstanceOf(HTMLDivElement);
      Array.from(div!.children).forEach((child, i) => {
        expect(child).toBeInstanceOf(HTMLDivElement);
        expect(child).toHaveClass('monospace-label');
        expect(child.textContent).toBe(expected[i]);
      });
    });
  });
});

describe('Test for editable data rendering', () => {
  function handler(e: Event) {}

  describe('Categroical input rendering test', () => {
    const vocab = ['contradiction', 'entailment', 'neutral'];

    it('renders a <select> with <options> for each vocab entry', () => {
      const formatted = renderCategoricalInput(vocab, handler);
      render(formatted, document.body);
      const select =
          document.body.querySelector<HTMLSelectElement>('select.dropdown');
      expect(select).toBeInstanceOf(HTMLSelectElement);
      expect(select!.value).toBeFalsy();
      Array.from(select!.options).forEach((option, i) => {
        expect(option).toBeInstanceOf(HTMLOptionElement);
        // The renderer adds a final option to the select representing a null
        // value, i.e., the empty item, thus this OR statement.
        const expected = vocab[i] || '';
        expect(option.textContent).toBe(expected);
        expect(option.value).toBe(expected);
      });
    });

    [...vocab, ''].forEach((value) => {
      it(
        `renders a <select> with the correct <option selected> for '${value}'`,
        () => {
          const formatted = renderCategoricalInput(vocab, handler, value);
          render(formatted, document.body);

          const select =
              document.body.querySelector<HTMLSelectElement>('select.dropdown');
          expect(select!.value).toBe(value);

          const option = select!.options[select!.selectedIndex];
          expect(option.textContent).toBe(value);
          expect(option.value).toBe(value);
        });
    });
  });

  describe('Long-form text input rendering test', () => {
    it('renders a textarea', () => {
      const formatted = renderTextInputLong(handler);
      render(formatted, document.body);
      const textarea =
        document.body.querySelector<HTMLTextAreaElement>('textarea.input-box');
      expect(textarea).toBeInstanceOf(HTMLTextAreaElement);
      expect(textarea!.value).toBeFalsy();
    });

    it('prefills a value in the rendered textarea', () => {
      const value = 'long text area input for tetsing what a long sentence.';
      const formatted = renderTextInputLong(handler, value);
      render(formatted, document.body);
      const textarea =
        document.body.querySelector<HTMLTextAreaElement>('textarea.input-box')!;
      expect(textarea.value).toBe(value);
    });

    it('styles the rendered textarea', () => {
      const formatted = renderTextInputLong(handler, '', {'color': 'red'});
      render(formatted, document.body);
      const textarea =
        document.body.querySelector<HTMLTextAreaElement>('textarea.input-box')!;
      expect(textarea.style.getPropertyValue('color')).toBe('red');
    });
  });

  describe('Short-form text input rendering test', () => {
    it('renders an input[type="text]', () => {
      const formatted = renderTextInputShort(handler);
      render(formatted, document.body);
      const input =
          document.body.querySelector<HTMLInputElement>('input.input-short');
      expect(input).toBeInstanceOf(HTMLInputElement);
      expect(input!.type).toBe('text');
      expect(input!.value).toBeFalsy();
    });

    it('prefills a value in the rendered input[type="text]', () => {
      const value = 'short text value';
      const formatted = renderTextInputShort(handler, value);
      render(formatted, document.body);
      const input =
          document.body.querySelector<HTMLInputElement>('input.input-short')!;
      expect(input.value).toBe(value);
    });
  });
});
