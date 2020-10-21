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

import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {observable, reaction} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {IndexedInput, ModelsMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';

import {styles} from './generated_text_module.css';
import {styles as sharedStyles} from './shared_styles.css';


// tslint:disable-next-line:no-any difflib does not support Closure imports


interface GeneratedTextResult {
  [key: string]: string;
}

interface TextDiff {
  inputStrings: string[];
  outputStrings: string[];
  equal: boolean[];
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

  @observable private generatedText: GeneratedTextResult[] = [];
  @observable private isChecked: boolean = true;

  firstUpdated() {
    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(
        getPrimarySelectedInputData, primarySelectedInputData => {
          if (primarySelectedInputData != null) {
            this.updateSelection([primarySelectedInputData]);
          }
        });
  }

  private async updateSelection(selectedInputData: IndexedInput[]) {
    this.generatedText = [];
    if (selectedInputData == null) {
      return;
    }

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        selectedInputData, this.model, dataset, ['GeneratedText'],
        'Generating text');
    const results = await this.loadLatest('generatedText', promise);
    if (results === null) return;

    this.generatedText = results;
  }

  render() {
    // Get the input field for the target text using the parent pointer of the
    // translation output field.
    const spec = this.appState.getModelSpec(this.model);
    const textKeys = findSpecKeys(spec.output, 'GeneratedText');
    const targetFields = textKeys.map(textKey => spec.output[textKey].parent!);

    const primarySelectedInputData =
        this.selectionService.primarySelectedInputData;
    if (primarySelectedInputData == null) {
      return;
    }
    const inputs = [primarySelectedInputData];

    const change = (e: Event) => {
      this.isChecked = !this.isChecked;
    };

    // tslint:disable-next-line:no-any
    // clang-format off
    return html`
      <div>
        <div class="checkbox-container ${inputs.length > 0 ? '': 'hidden'}">
          <lit-checkbox ?checked=${this.isChecked} @change=${change}></lit-checkbox>
          <div>Display word-wise diffs</div>
        </div>
        <div class="results-holder">
          ${this.generatedText.map(
            (output, i) => this.renderOutput(output, targetFields, inputs[i]))}
        </div>
      </div>
    `;
    // clang-format on
  }

  renderOutput(
      output: GeneratedTextResult, targetFields: Array<string|null>,
      input: IndexedInput) {
    const keys = Object.keys(output);

    return keys.map((key, i) => {
      const targetField = targetFields[i];
      if (targetField == null) {
        return this.renderTranslationText(key, output[key]);
      } else {
        const targetText = input.data[targetField];
        return this.renderDiffText(targetField, targetText, key, output[key]);
      }
    });
  }

  renderTranslationText(key: string, text: string) {
    return html`
      <div class="output">
        <div class="key">${key} </div>
        <div class="value">${text}</div>
      </div>
    `;
  }

  renderDiffText(
      targetField: string, targetText: string, outputKey: string,
      outputText: string) {
    return html`
      <lit-text-diff
        beforeLabel=${targetField}
        beforeText=${targetText}        
        afterLabel=${outputKey}
        afterText=${outputText}
      /></lit-text-diff>
    `;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, 'GeneratedText');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generated-text-module': GeneratedTextModule;
  }
}
