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

import {canonicalizeGenerationResults, getAllOutputTexts, getAllReferenceTexts, getFlatTexts} from './generated_text_utils';
import {Input, LitType, Preds, Spec} from './types';

// TODO(b/162269499): replace with a real class constructor.
function textSegmentType(): LitType {
  return {
    '__class__': 'LitType',
    '__name__': 'TextSegment',
    '__mro__': ['TextSegment', 'LitType', 'object'],
    'required': false,
  };
}

// TODO(b/162269499): replace with a real class constructor.
function referenceTextsType(): LitType {
  return {
    '__class__': 'LitType',
    '__name__': 'ReferenceTexts',
    '__mro__': ['ReferenceTexts', 'LitType', 'object'],
    'required': false,
  };
}

// TODO(b/162269499): replace with a real class constructor.
function generatedTextType(parent: string): LitType {
  return {
    '__class__': 'LitType',
    '__name__': 'GeneratedText',
    '__mro__': ['GeneratedText', 'TextSegment', 'LitType', 'object'],
    'required': false,
    parent
  };
}

// TODO(b/162269499): replace with a real class constructor.
function generatedTextCandidatesType(parent: string): LitType {
  return {
    '__class__': 'LitType',
    '__name__': 'GeneratedTextCandidates',
    '__mro__': ['GeneratedTextCandidates', 'TextSegment', 'LitType', 'object'],
    'required': false,
    parent
  };
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
