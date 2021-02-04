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

// tslint:disable:no-new-decorators
import difflib from 'difflib';

import {customElement, html, TemplateResult} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';

import {styles} from './generated_text_module.css';
import {styles as sharedStyles} from './shared_styles.css';


// tslint:disable-next-line:no-any difflib does not support Closure imports
// difflib declare placeholder - DO NOT REMOVE

interface GeneratedTextResult {
  [key: string]: string;
}

interface TextDiff {
  inputStrings: string[];
  outputStrings: string[];
  equal: boolean[];
}

enum DiffMode {
  NONE = 'None',
  WORD = 'Word',
  CHAR = 'Character',
}

/**
 * A LIT module that renders generated text.
 */
@customElement('generated-text-module')
export class GeneratedTextModule extends LitModule {
  static title = 'Generated Text';
  static template = (model = '') => {
    return html`
      <generated-text-module model=${model}>
      </generated-text-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  @observable private generatedText: GeneratedTextResult = {};
  @observable private diffMode: DiffMode = DiffMode.NONE;

  firstUpdated() {
    this.reactImmediately(
        () => this.selectionService.primarySelectedInputData, data => {
          this.updateSelection(data);
        });
  }

  private async updateSelection(inputData: IndexedInput|null) {
    this.generatedText = {};
    if (inputData == null) return;

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [inputData], this.model, dataset, ['GeneratedText'], 'Generating text');
    const results = await this.loadLatest('generatedText', promise);
    if (results === null) return;

    this.generatedText = results[0];
  }

  render() {
    // Get the input field for the target text using the parent pointer of the
    // translation output field.
    const spec = this.appState.getModelSpec(this.model);
    const textKeys = findSpecKeys(spec.output, 'GeneratedText');
    const targetFields = textKeys.map(textKey => spec.output[textKey].parent!);

    const inputData = this.selectionService.primarySelectedInputData;
    if (inputData == null) return;

    const setDiffMode = (e: Event) => {
      this.diffMode = (e.target as HTMLInputElement).value as DiffMode;
    };

    // clang-format off
    return html`
      <div>
        <div class="diff-selector">
          Highlight diffs:
          ${Object.values(DiffMode).map(val => html`
            <input type="radio" name="diffSelect" value="${val}" id='diff${val}'
             ?checked=${val === this.diffMode} @change=${setDiffMode}>
            <label for='diff${val}'>${val}</label>
          `)}
        </div>
        <div class="results-holder">
          ${this.renderOutput(this.generatedText, targetFields, inputData)}
        </div>
      </div>
    `;
    // clang-format on
  }

  renderTextField(name: string, text: string|TemplateResult) {
    // clang-format off
    return html`
      <div class="output">
        <div class="key">${name} </div>
        <div class="value">${text}</div>
      </div>
    `;
    // clang-format on
  }

  renderOutput(
      output: GeneratedTextResult, targetFields: Array<string|null>,
      input: IndexedInput) {
    return Object.keys(output).map((key, i) => {
      const targetField = targetFields[i];
      if (targetField == null) {
        // Single field; no reference.
        return this.renderTextField(key, output[key]);
      } else if (this.diffMode === 'None') {
        // Output and reference; no diffs.
        const targetText = input.data[targetField];
        // clang-format off
        return html`
          ${this.renderTextField(targetField, targetText)}
          ${this.renderTextField(key, output[key])}
          <br>
        `;
        // clang-format on
      } else {
        // Output and reference; compute diffs.
        const targetText = input.data[targetField];
        const byWord = (this.diffMode === 'Word');
        const textDiff: TextDiff = getTextDiff(targetText, output[key], byWord);
        // clang-format off
        return html`
          ${this.renderDiffString(
              targetField, textDiff.inputStrings, textDiff.equal, byWord)}
          ${this.renderDiffString(
              key, textDiff.outputStrings, textDiff.equal, byWord)}
          <br>
        `;
        // clang-format on
      }
    });
  }

  renderDiffString(
      key: string, strings: string[], equal: boolean[], byWord: boolean) {
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

    const displaySpans = displayStrings.map((output, i) => {
      const classes = classMap({highlighted: !equal[i]});
      return html`<span class=${classes}>${output}</span>`;
    });
    return this.renderTextField(key, html`${displaySpans}`);
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'GeneratedText');
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
    'generated-text-module': GeneratedTextModule;
  }
}
