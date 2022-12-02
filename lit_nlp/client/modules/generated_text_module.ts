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
import '@material/mwc-switch';
import '../elements/generated_text_vis';

import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {styles as visStyles} from '../elements/generated_text_vis.css';
import {DiffMode, GeneratedTextResult, GENERATION_TYPES} from '../lib/generated_text_utils';
import {GeneratedText, GeneratedTextCandidates, LitTypeWithParent, ReferenceScores, ReferenceTexts} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, Input, ModelInfoMap, Spec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys} from '../lib/utils';

import {styles} from './generated_text_module.css';

interface ReferenceScoresResult {
  [referenceFieldName: string]: number[];
}

interface OutputGroupKeys {
  reference?: string;            /* field of this.inputData */
  generated?: string;            /* field of this.generatedText */
  referenceModelScores?: string; /* field of this.referenceScores */
}

/**
 * A LIT module that renders generated text.
 */
@customElement('generated-text-module')
export class GeneratedTextModule extends LitModule {
  static override title = 'Generated Text';
  static override duplicateForExampleComparison = true;
  static override duplicateAsRow = false;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <generated-text-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </generated-text-module>`;

  static override get styles() {
    return [sharedStyles, visStyles, styles];
  }

  @observable private inputData: Input|null = null;
  @observable private generatedText: GeneratedTextResult = {};
  @observable private referenceScores: ReferenceScoresResult = {};
  @observable private diffMode: DiffMode = DiffMode.NONE;
  @observable private invertDiffs: boolean = false;

  @computed
  get referenceFields(): Map<string, string> {
    const dataSpec = this.appState.currentDatasetSpec;
    const outputSpec = this.appState.getModelSpec(this.model).output;
    const refMap = new Map<string, string>();
    const textKeys = findSpecKeys(outputSpec, GENERATION_TYPES);
    for (const textKey of textKeys) {
      const {parent} = outputSpec[textKey] as LitTypeWithParent;
      if (parent && dataSpec[parent]) {
        refMap.set(textKey, parent);
      }
    }
    return refMap;
  }

  @computed
  get referenceScoreFields(): Map<string, string> {
    // Map of output field -> input field
    const dataSpec = this.appState.currentDatasetSpec;
    const outputSpec = this.appState.getModelSpec(this.model).output;
    const refMap = new Map<string, string>();
    const scoreKeys = findSpecKeys(outputSpec, ReferenceScores);
    for (const scoreKey of scoreKeys) {
      const parent = (outputSpec[scoreKey] as ReferenceScores).parent;
      if (parent && dataSpec[parent]) {
        refMap.set(scoreKey, parent);
      }
    }
    return refMap;
  }

  override firstUpdated() {
    this.reactImmediately(
        () => this.selectionService.primarySelectedInputData, data => {
          this.updateSelection(data);
        });
  }

  private async updateSelection(input: IndexedInput|null) {
    this.inputData = null;
    this.generatedText = {};
    this.referenceScores = {};
    if (input == null) return;

    const dataset = this.appState.currentDataset;
    const promise = this.apiService.getPreds(
        [input], this.model, dataset, [...GENERATION_TYPES, ReferenceScores],
        [], 'Generating text');
    const results = await this.loadLatest('generatedText', promise);
    if (results === null) return;

    this.inputData = input.data;
    // Post-process results
    const spec = this.appState.getModelSpec(this.model).output;
    const result = results[0];
    for (const key of Object.keys(result)) {
      if (spec[key] instanceof GeneratedText) {
        this.generatedText[key] = [[result[key], null]];
      }
      if (spec[key] instanceof GeneratedTextCandidates) {
        this.generatedText[key] = result[key];
      }
      if (spec[key] instanceof ReferenceScores) {
        const referenceFieldName = this.referenceScoreFields.get(key);
        if (referenceFieldName != null) {
          this.referenceScores[referenceFieldName] = result[key];
        } else {
          console.log(`Ignoring ReferenceScores field '${
              key}' found with no parent in the dataset.`);
        }
      }
    }
  }

  renderToolbar() {
    if (!this.referenceFields.size) {
      return null;
    }

    const isDiffActive = this.diffMode !== DiffMode.NONE;

    const setDiffMode = (e: Event) => {
      this.diffMode = (e.target as HTMLInputElement).value as DiffMode;
    };

    const toggleInvertDiffs = () => {
      this.invertDiffs = !this.invertDiffs;
    };

    // clang-format off
    return html`
      <div class='module-toolbar'>
        <div class="diff-selector">
          Highlight comparison:
          ${Object.values(DiffMode).map(val => html`
            <input type="radio" name="diffSelect" value="${val}" id='diff${val}'
             ?checked=${val === this.diffMode} @change=${setDiffMode}>
            <label for='diff${val}'>${val}</label>
          `)}
        </div>
        <div class=${classMap({'switch-container': true,
                               'switch-container-disabled': !isDiffActive})}
          @click=${isDiffActive ? toggleInvertDiffs : null}>
          <div class=${isDiffActive && !this.invertDiffs ? 'highlighted-diff' : ''}>
            Diffs
          </div>
          <mwc-switch ?selected=${this.invertDiffs} ?disabled=${!isDiffActive}>
          </mwc-switch>
          <div class=${isDiffActive && this.invertDiffs ? 'highlighted-match' : ''}>
            Matches
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderOutputGroup(g: OutputGroupKeys) {
    const candidates =
        g.generated != null ? this.generatedText[g.generated] : [];

    const referenceFieldName = g.reference;
    let referenceTexts =
        referenceFieldName != null ? this.inputData?.[referenceFieldName] : [];
    // If the reference is a TextSegment, up-cast the single string to the
    // expected candidate list type GenereatedTextCandidate[].
    if (referenceFieldName !== undefined) {
      const spec = this.appState.getModelSpec(this.model).input;
      if (!(spec[referenceFieldName] instanceof ReferenceTexts)) {
        referenceTexts = [[referenceTexts, null]];
      }
    }

    const referenceModelScores = g.referenceModelScores != null ?
        this.referenceScores[g.referenceModelScores] :
        [];

    // clang-format off
    return html`
      <generated-text-vis .fieldName=${g.generated}
                          .candidates=${candidates}
                          .referenceFieldName=${referenceFieldName}
                          .referenceTexts=${referenceTexts}
                          .referenceModelScores=${referenceModelScores}
                          .diffMode=${this.diffMode}
                          ?highlightMatch=${this.invertDiffs}>
      </generated-text-vis>
    `;
    // clang-format on
  }

  override renderImpl() {
    // Create output groups, making sure all fields are represented.
    const scoreFieldSet = new Set<string>(Object.keys(this.referenceScores));
    const outputGroups: OutputGroupKeys[] = [];
    for (const genFieldName of Object.keys(this.generatedText)) {
      const outputGroup: OutputGroupKeys = {
        generated: genFieldName,
        reference: this.referenceFields.get(genFieldName)
      };
      if (outputGroup.reference != null &&
          this.referenceScores[outputGroup.reference] != null) {
        outputGroup.referenceModelScores = outputGroup.reference;
        scoreFieldSet.delete(outputGroup.reference);
      }
      outputGroups.push(outputGroup);
    }
    for (const scoreFieldName of scoreFieldSet) {
      outputGroups.push(
          {reference: scoreFieldName, referenceModelScores: scoreFieldName});
    }

    // clang-format off
    return html`
      <div class='module-container'>
        ${this.renderToolbar()}
        <div class="module-results-area">
          ${outputGroups.map(g => this.renderOutputGroup(g))}
        </div>
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, GENERATION_TYPES);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generated-text-module': GeneratedTextModule;
  }
}
