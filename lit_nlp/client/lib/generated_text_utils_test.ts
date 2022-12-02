/**
 * @license
 * Copyright 2022 Google LLC
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

import 'jasmine';

import {canonicalizeGenerationResults, getAllOutputTexts, getAllReferenceTexts, getFlatTexts, getTextDiff} from './generated_text_utils';
import {GeneratedText, GeneratedTextCandidates, LitType, ReferenceTexts, TextSegment} from './lit_types';
import {Input, Preds, Spec} from './types';
import {createLitType} from './utils';

function textSegmentType(): LitType {
  return createLitType(TextSegment, {
    'required': false,
  });
}

function referenceTextsType(): LitType {
  return createLitType(ReferenceTexts, {
    'required': false,
  });
}

function generatedTextType(parent: string): LitType {
  return createLitType(GeneratedText, {'required': false, 'parent': parent});
}

function generatedTextCandidatesType(parent: string): LitType {
  return createLitType(
      GeneratedTextCandidates, {'required': false, 'parent': parent});
}

describe('canonicalizeGenerationResults test', () => {
  it('upcasts GeneratedText', () => {
    const outputSpec:
        Spec = {'generated_text': generatedTextType('input_text')};

    const rawPreds: Preds = {'generated_text': 'foo'};

    const expected: Preds = {'generated_text': [['foo', null]]};

    expect(canonicalizeGenerationResults(rawPreds, outputSpec))
        .toEqual(expected);
  });

  it('preserves GeneratedTextCandidates', () => {
    const outputSpec: Spec = {
      'generated_candidates': generatedTextCandidatesType('input_text')
    };

    const rawPreds:
        Preds = {'generated_candidates': [['foo', -0.1], ['bar', -0.5]]};

    expect(canonicalizeGenerationResults(rawPreds, outputSpec))
        .toEqual(rawPreds);
  });

  it('handles mixed types', () => {
    const outputSpec: Spec = {
      'generated_text': generatedTextType('input_text'),
      'generated_candidates': generatedTextCandidatesType('input_text')
    };

    const rawPreds: Preds = {
      'generated_text': 'foo',
      'generated_candidates': [['foo', -0.1], ['bar', -0.5]]
    };

    const expected: Preds = {
      'generated_text': [['foo', null]],
      'generated_candidates': [['foo', -0.1], ['bar', -0.5]]
    };

    expect(canonicalizeGenerationResults(rawPreds, outputSpec))
        .toEqual(expected);
  });
});

describe('getFlatTexts test', () => {
  it('works for GeneratedText', () => {
    const preds:
        Preds = {'gen_one': 'foo', 'skipped': 'spam', 'gen_two': 'bar'};

    expect(getFlatTexts(['gen_one', 'gen_two'], preds)).toEqual(['foo', 'bar']);
  });

  it('works for GeneratedTextCandidates', () => {
    const preds:
        Preds = {'single': 'foo', 'candidates': [['bar', -0.1], ['baz', -0.5]]};

    expect(getFlatTexts(['candidates'], preds)).toEqual(['bar', 'baz']);
    expect(getFlatTexts(['single', 'candidates'], preds)).toEqual([
      'foo', 'bar', 'baz'
    ]);
  });
});

describe('getAllReferenceTexts test', () => {
  it('works for multiple output fields', () => {
    const dataSpec: Spec = {
      'single_reference': textSegmentType(),
      'unused_reference': textSegmentType()
    };

    const outputSpec: Spec = {
      'generated_text': generatedTextType('single_reference'),
      'generated_candidates': generatedTextCandidatesType('single_reference')
    };

    const input: Input = {'single_reference': 'foo', 'unused_reference': 'bar'};
    const indexedInput = {id: '123', data: input, meta: {}};

    // Output should include the reference only once, even though two output
    // fields refer to it.
    expect(getAllReferenceTexts(dataSpec, outputSpec, indexedInput)).toEqual([
      'foo'
    ]);
  });

  it('works for ReferenceTexts', () => {
    const dataSpec: Spec = {
      'single_reference': textSegmentType(),
      'multi_reference': referenceTextsType(),
    };

    const outputSpec: Spec = {
      'generated_text': generatedTextType('single_reference'),
      'generated_candidates': generatedTextCandidatesType('multi_reference')
    };

    const input: Input = {
      'single_reference': 'foo',
      'multi_reference': [['bar', -0.1], ['baz', -0.5]]
    };
    const indexedInput = {id: '123', data: input, meta: {}};

    expect(getAllReferenceTexts(dataSpec, outputSpec, indexedInput)).toEqual([
      'foo', 'bar', 'baz'
    ]);
  });
});

describe('getAllOutputTexts test', () => {
  it('works for mixed types', () => {
    const outputSpec: Spec = {
      'single': generatedTextType(''),
      'candidates': generatedTextCandidatesType(''),
      'unsupported_type': textSegmentType()
    };

    const preds: Preds = {
      'single': 'foo',
      'candidates': [['bar', -0.1], ['baz', -0.5]],
      'unsupported_type': 'not_generation_output'
    };

    expect(getAllOutputTexts(outputSpec, preds)).toEqual(['foo', 'bar', 'baz']);
  });
});

describe('getTextDiff test', () => {
  /** Character diff tests */
  it('gets character text diffs for empty strings', () => {
    const diff = getTextDiff('', '', false);
    expect(diff.inputStrings).toEqual([]);
    expect(diff.outputStrings).toEqual([]);
    expect(diff.equal).toEqual([]);
  });

  it('gets character text diffs for different lengths', () => {
    let diff = getTextDiff('', 'a', false);
    expect(diff.inputStrings).toEqual(['']);
    expect(diff.outputStrings).toEqual(['a']);
    expect(diff.equal).toEqual([false]);

    diff = getTextDiff('a', 'ab', false);
    expect(diff.inputStrings).toEqual(['a', '']);
    expect(diff.outputStrings).toEqual(['a', 'b']);
    expect(diff.equal).toEqual([true, false]);
  });

  it('gets character text diffs for single character', () => {
    // Matching characters.
    let diff = getTextDiff('a', 'a', false);
    expect(diff.inputStrings).toEqual(['a']);
    expect(diff.outputStrings).toEqual(['a']);
    expect(diff.equal).toEqual([true]);

    // Non-matching characters.
    diff = getTextDiff('a', 'b', false);
    expect(diff.inputStrings).toEqual(['a']);
    expect(diff.outputStrings).toEqual(['b']);
    expect(diff.equal).toEqual([false]);
  });

  it('gets character text diffs for multiple characters', () => {
    // All characters match.
    let diff = getTextDiff('aba', 'aba', false);
    expect(diff.inputStrings).toEqual(['aba']);
    expect(diff.outputStrings).toEqual(['aba']);
    expect(diff.equal).toEqual([true]);

    // Some characters match.
    diff = getTextDiff('cba', 'aba', false);
    expect(diff.inputStrings).toEqual(['c', 'ba']);
    expect(diff.outputStrings).toEqual(['a', 'ba']);
    expect(diff.equal).toEqual([false, true]);

    diff = getTextDiff('abc', 'cde', false);
    expect(diff.inputStrings).toEqual(['ab', 'c', '']);
    expect(diff.outputStrings).toEqual(['', 'c', 'de']);
    expect(diff.equal).toEqual([false, true, false]);

    // No characters match.
    diff = getTextDiff('abc', 'def', false);
    expect(diff.inputStrings).toEqual(['abc']);
    expect(diff.outputStrings).toEqual(['def']);
    expect(diff.equal).toEqual([false]);
  });

  it('gets character text diffs for multiple words', () => {
    const diff = getTextDiff('I went there', 'I want this', false);
    expect(diff.inputStrings).toEqual(['I w', 'e', 'nt th', 'ere']);
    expect(diff.outputStrings).toEqual(['I w', 'a', 'nt th', 'is']);
    expect(diff.equal).toEqual([true, false, true, false]);
  });

  /** Word diff tests */
  it('gets word text diffs for empty strings', () => {
    const diff = getTextDiff('', '', true);
    expect(diff.inputStrings).toEqual(['']);
    expect(diff.outputStrings).toEqual(['']);
    expect(diff.equal).toEqual([true]);
  });

  it('gets word text diffs for single word', () => {
    let diff = getTextDiff('ab', 'ab', true);
    expect(diff.inputStrings).toEqual(['ab']);
    expect(diff.outputStrings).toEqual(['ab']);
    expect(diff.equal).toEqual([true]);

    diff = getTextDiff('ab', 'ba', true);
    expect(diff.inputStrings).toEqual(['ab']);
    expect(diff.outputStrings).toEqual(['ba']);
    expect(diff.equal).toEqual([false]);
  });

  it('gets word text diffs for different lengths', () => {
    const diff = getTextDiff('I went there', 'They went', true);
    expect(diff.inputStrings).toEqual(['I', 'went', 'there']);
    expect(diff.outputStrings).toEqual(['They', 'went', '']);
    expect(diff.equal).toEqual([false, true, false]);
  });

  it('gets word text diffs for multiple words', () => {
    // All words match.
    let diff = getTextDiff('I went there', 'I went there', true);
    expect(diff.inputStrings).toEqual(['I went there']);
    expect(diff.outputStrings).toEqual(['I went there']);
    expect(diff.equal).toEqual([true]);

    // Some words match.
    diff = getTextDiff('I went there already', 'I want this already', true);
    expect(diff.inputStrings).toEqual(['I', 'went there', 'already']);
    expect(diff.outputStrings).toEqual(['I', 'want this', 'already']);
    expect(diff.equal).toEqual([true, false, true]);

    // No words match
    diff = getTextDiff('I went there', 'He wants this', true);
    expect(diff.inputStrings).toEqual(['I went there']);
    expect(diff.outputStrings).toEqual(['He wants this']);
    expect(diff.equal).toEqual([false]);
  });
});
