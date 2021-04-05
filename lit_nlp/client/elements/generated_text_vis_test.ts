/**
 * @license
 * Copyright 2020 Google LLC
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
import {getTextDiff} from './generated_text_vis';

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
