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

import difflib from 'difflib';

import {css, customElement, html, LitElement, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';

import {styles} from './text_diff.css';

interface TextDiff {
  inputStrings: string[];
  outputStrings: string[];
  equal: boolean[];
}


/**
 * A component that renders the diff between two strings, with the differences
 * highlighted.  It can show before before and after, or only after.
 */
@customElement('lit-text-diff')
export class TextDiffComponent extends LitElement {
  /* The two text values to diff */
  @property({type: String}) beforeText = undefined;
  @property({type: String}) afterText = undefined;

  /* Show a label next to the before/after text itself */
  @property({type: String}) beforeLabel = undefined;
  @property({type: String}) afterLabel = undefined;

  /* Allow only rendering before or after, instead of both */
  @property({type: Boolean}) includeBefore = true;
  @property({type: Boolean}) includeAfter = true;

  /* Optional: Diff by character instead of by word */
  @property({type: Boolean}) byCharacter = false;

  static get styles() {
    return styles;
  }

  render() {
    // textDiff contains arrays of parsed segments from both strings, and an
    // array of booleans indicating whether the corresponding change type is
    // 'equal.'
    const textDiff = getTextDiff(this.beforeText!, this.afterText!, !this.byCharacter);


    // Highlight strings as bold if they don't match (and the changetype is
    // not 'equal').
    // clang-format off
    return html`
      ${this.includeBefore
        ? this.renderDiffString(textDiff.inputStrings, textDiff.equal, this.beforeLabel)
        : null}
      ${this.includeAfter
        ? this.renderDiffString(textDiff.outputStrings, textDiff.equal, this.afterLabel)
        : null}
      <br>
    `;
    // clang-format on
  }

  renderDiffString(strings: string[], equal: boolean[], label?: string) {
    const byWord = !this.byCharacter;
    let displayStrings = strings;

    // Add spaces between strings for the word-wise character diffs.
    if (byWord) {
      const lastIndex = strings.length - 1;
      displayStrings = strings.map((item, i) => {
        if (i !== lastIndex) {
          return item.concat(' ');
        }
        return item;
      });
    }

    // clang-format off
    return html`
      <div class="output">
        ${label && html`<div class="key">${label} </div>`}
        <div class="value">
          ${displayStrings.map((output, i) => {
            const classes = classMap({
              highlighted: !equal[i],
            });
            return html`<span class=${classes}>${output}</span>`;
          })}
        </div>
      </div>
    `;
    // clang-format on
  }
}


/**
 * Uses difflib library to compute character differences between the input
 * strings and returns a TextDiff object, which contains arrays of parsed
 * segments from both strings and an array of booleans indicating whether the
 * corresponding change type is 'equal.'
 */
export function getTextDiff(
    targetText: string, outputText: string, byWord: boolean): TextDiff {
  // Use difflib library to compute opcodes, which contain a group of changes
  // between the two input strings. Each opcode contains the change type and
  // the start/end of the concerned characters/words in each string.
  const targetWords = targetText.split(' ');
  const outputWords = outputText.split(' ');

  const matcher = byWord ?
      new difflib.SequenceMatcher(() => false, targetWords, outputWords) :
      new difflib.SequenceMatcher(() => false, targetText, outputText);
  const opcodes = matcher.getOpcodes();

  // Store an array of the parsed segments from both strings and whether
  // the change type is 'equal.'
  const inputStrings: string[] = [];
  const outputStrings: string[] = [];
  const equal: boolean[] = [];

  for (const opcode of opcodes) {
    const changeType = opcode[0];
    const startA = Number(opcode[1]);
    const endA = Number(opcode[2]);
    const startB = Number(opcode[3]);
    const endB = Number(opcode[4]);

    equal.push((changeType === 'equal'));

    if (byWord) {
      inputStrings.push(targetWords.slice(startA, endA).join(' '));
      outputStrings.push(outputWords.slice(startB, endB).join(' '));
    } else {
      inputStrings.push(targetText.slice(startA, endA));
      outputStrings.push(outputText.slice(startB, endB));
    }
  }

  const textDiff: TextDiff = {inputStrings, outputStrings, equal};
  return textDiff;
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-text-diff': TextDiffComponent;
  }
}
