/**
 * @license
 * Copyright 2021 Google LLC
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
import {customElement} from 'lit/decorators';
import {css, html} from 'lit';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, LitName, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain} from '../lib/utils';

/**
 * A LIT module that renders generated text.
 */
@customElement('generated-image-module')
export class GeneratedImageModule extends LitModule {
  static override title = 'Generated Images';
  static override duplicateForExampleComparison = true;
  static override duplicateAsRow = true;
  static override template = (model = '', selectionServiceIndex = 0) => {
    return html`
      <generated-image-module model=${model}
        selectionServiceIndex=${selectionServiceIndex}>
      </generated-image-module>`;
  };


  static supportedTypes: LitName[] = ['ImageBytes'];

  static override get styles() {
    const styles = css`
      .field-group {
        padding: 4px;
      }
    `;
    return [sharedStyles, styles];
  }

  @observable private generatedImages: {[key: string]: string} = {};

  override firstUpdated() {
    this.reactImmediately(
        () => this.selectionService.primarySelectedInputData, data => {
          this.updateSelection(data);
        });
  }

  private async updateSelection(input: IndexedInput|null) {
    this.generatedImages = {};
    if (input == null) return;

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [input], this.model, dataset, GeneratedImageModule.supportedTypes,
        'Generating images');
    const results = await this.loadLatest('generatedImages', promise);
    if (results === null) return;

    this.generatedImages = results[0];
  }

  renderOutputGroup(name: string) {
    // clang-format off
    return html`
      <div class='field-group'>
        <div class="field-title">${name}</div>
        <img src=${this.generatedImages[name]}>
      </div>
    `;
    // clang-format on
  }

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class="module-results-area">
          ${Object.keys(this.generatedImages).map(
              name => this.renderOutputGroup(name))}
        </div>
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(
        modelSpecs, GeneratedImageModule.supportedTypes);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generated-image-module': GeneratedImageModule;
  }
}
