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

import '../elements/generator_controls';

// tslint:disable:no-new-decorators
import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {CallConfig, formatForDisplay, IndexedInput, Input, LitName, ModelInfoMap, Spec} from '../lib/types';
import {flatten, isLitSubtype} from '../lib/utils';
import {GroupService} from '../services/group_service';
import {SelectionService} from '../services/services';

import {styles} from './generator_module.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * A LIT module that allows the user to generate new examples.
 */
@customElement('generator-module')
export class GeneratorModule extends LitModule {
  static title = 'Datapoint Generator';
  static numCols = 10;

  static template = () => {
    return html`<generator-module></generator-module>`;
  };

  static duplicateForModelComparison = false;
  private readonly groupService = app.getService(GroupService);

  static get styles() {
    return [sharedStyles, styles];
  }

  @observable editedData: Input = {};
  @observable isGenerating = false;

  @observable generated: IndexedInput[][] = [];
  @observable appliedGenerator: string|null = null;


  @computed
  get datasetName() {
    return this.appState.currentDataset;
  }
  // TODO(lit-team): make model configurable.
  @computed
  get modelName() {
    return this.appState.currentModels[0];
  }

  @computed
  get totalNumGenerated() {
    return this.generated.reduce((a, b) => a + b.length, 0);
  }

  firstUpdated() {
    const getSelectedData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedData, selectedData => {
      if (this.selectionService.lastUser !== this) {
        this.resetEditedData();
      }
    });

    // If all staged examples are removed one-by-one, make sure we reset
    // to a clean state.
    this.react(() => this.totalNumGenerated, numAvailable => {
      if (numAvailable <= 0) {
        this.resetEditedData();
      }
    });
  }


  updated() {
    // Update the header items to be the width of the rows of the table.
    const header = this.shadowRoot!.getElementById('header') as ParentNode;
    const firstRow = this.shadowRoot!.querySelector('.row') as ParentNode;
    if (header) {
      for (let i = 0; i < header.children.length; i++) {
        const width = (firstRow.children[i] as HTMLElement).offsetWidth;
        const child = header.children[i];
        (child as HTMLElement).style.minWidth = `${width}px`;
      }
    }

    // Add event listeners for generation events from individual generators.
    const onGenClick = (event: Event) => {
      const globalParams = {
        'model_name': this.modelName,
        'dataset_name': this.datasetName,
      };
      // tslint:disable-next-line:no-any
      const generatorParams: {[setting: string]: string} = (event as any)
          .detail.settings;
      // tslint:disable-next-line:no-any
      const generatorName =  (event as any).detail.name;

      // Add user-specified parameters from the applied generator.
      const allParams = Object.assign({}, globalParams, generatorParams);
      this.handleGeneratorClick(generatorName, allParams);
    };
    const controls =
        this.shadowRoot!.querySelectorAll('lit-generator-controls');
    for (let i = 0; i < controls.length; i++) {
      controls[i].addEventListener('generator-click', onGenClick);
    }
  }

  private resetEditedData() {
    this.generated = [];
    this.appliedGenerator = null;
  }

  private handleGeneratorClick(generator: string, config?: CallConfig) {
    if (!this.isGenerating) {
      this.generate(generator, this.modelName, config);
    }
  }

  private async generate(
      generator: string, modelName: string, config?: CallConfig) {
    this.isGenerating = true;
    const sourceExamples = this.selectionService.selectedOrAllInputData;
    try {
      const generated = await this.apiService.getGenerated(
          sourceExamples, modelName, this.appState.currentDataset, generator,
          config);
      // Populate additional metadata fields.
      // parentId and source should already be set from the backend.
      for (const examples of generated) {
        for (const ex of examples) {
          Object.assign(ex['meta'], {added: 1});
        }
      }
      this.generated = generated;
      this.appliedGenerator = generator;
      this.isGenerating = false;
    } catch (err) {
      this.isGenerating = false;
    }
  }

  private async createNewDatapoints(data: IndexedInput[][]) {
    const newExamples = flatten(data);
    this.appState.commitNewDatapoints(newExamples);
    const newIds = newExamples.map(d => d.id);
    if (newIds.length === 0) return;

    const parentIds =
        new Set<string>(newExamples.map(ex => ex.meta['parentId']));

    // Select parents and children, and set primary to the first child.
    this.selectionService.selectIds([...parentIds, ...newIds], this);
    this.selectionService.setPrimarySelection(newIds[0], this);

    // If in comparison mode, set reference selection to the parent point
    // for direct comparison.
    if (this.appState.compareExamplesEnabled) {
      const referenceSelectionService =
          app.getServiceArray(SelectionService)[1];
      referenceSelectionService.selectIds([...parentIds, ...newIds], this);
      // parentIds[0] is not necessarily the parent of newIds[0], if
      // generated[0] is [].
      const parentId = newExamples[0].meta['parentId'];
      referenceSelectionService.setPrimarySelection(parentId, this);
    }
  }

  render() {
    return html`
      <div class="generator-module-wrapper">
        ${this.renderGeneratorButtons()}
        ${this.renderGenerated()}
      </div>
    `;
  }

  renderGenerated() {
    const isGenerating = this.isGenerating;
    const nothingGenerated =
        this.appliedGenerator !== null && this.totalNumGenerated === 0;

    // clang-format off
    return html`
        <div id='generated-holder'>
          ${this.appliedGenerator === null || isGenerating || nothingGenerated ?
            null :
            html`
              <div class="counterfactuals-count">
                Generated ${this.totalNumGenerated}
                ${this.totalNumGenerated === 1 ?
                'counterfactual' : 'counterfactuals'}
                from ${this.appliedGenerator}.
              </div>
            `}
          ${this.renderHeader()}
          <div class="entries">
            ${isGenerating ? html`<div>Generating...</div>` : null}
            ${nothingGenerated ?
                html`<div>Nothing available for this generator</div>` :
                null}
            ${this.renderEntries()}
        </div>
      </div>
    `;
    // clang-format on
  }

  /**
   * Render the generated counterfactuals themselves.
   */
  renderEntries() {
    return this.generated.map((generatedList, parentIndex) => {
      return generatedList.map((generated, generatedIndex) => {
        const addPoint = async () => {
          this.generated[parentIndex].splice(generatedIndex, 1);
          await this.createNewDatapoints([[generated]]);
        };
        const removePoint = () => {
          this.generated[parentIndex].splice(generatedIndex, 1);
        };
        const fieldNames = Object.keys(generated.data);

        // render values for each datapoint.
        // clang-format off
        return html`
          <div class='row'>
            ${fieldNames.map((key) => {
              const editable =
                  !this.appState.currentModelRequiredInputSpecKeys.includes(
                      key);
              return this.renderEntry(generated.data, key, editable);
            })}
            <button class="button add-button" @click=${addPoint}>Add</button>
            <button class="button" @click=${removePoint}>Remove</button>
          </div>
        `;
        // clang-format on
      });
    });
  }

  renderHeader() {
    const onAddAll = async () => {
      await this.createNewDatapoints(this.generated);
      this.resetEditedData();
    };

    if (this.totalNumGenerated <= 0) {
      return null;
    }
    // clang-format off
    return html`
      <div id='header'>
        ${this.renderKeys()}
        ${this.totalNumGenerated <= 0 ? null : html`
          <button class='button add-button' @click=${onAddAll}>
             Add all
          </button>
          <button class='button' @click=${this.resetEditedData}>
             Clear
          </button>
        `}
      </div>`;
    // clang-format on
  }

  renderKeys() {
    const keys = this.appState.currentInputDataKeys;
    // clang-format off
    return html`${keys.map(key => html`<div class='entry'>${key}</div>`)}`;
    // clang-format on
  }

  renderGeneratorButtons() {
    const data = this.selectionService.selectedOrAllInputData;
    const generatorsInfo = this.appState.metadata.generators;
    const generators = Object.keys(generatorsInfo);
    const text = generators.length > 0 ?
        `Generate counterfactuals for current selection (${
            data.length} datapoint${data.length === 1 ? '' : `s`}):` :
        'No generators provided by the server.';

    // clang-format off
    return html`
        <div id="generators">
          <div>${text}</div>
          ${generators.map(genName => {
            const spec = generatorsInfo[genName].spec;
            Object.keys(spec).forEach(fieldName => {
              // If the generator uses a field matcher, then get the matching
              // field names from the specified spec and use them as the vocab.
              if (isLitSubtype(spec[fieldName], 'FieldMatcher')) {
                spec[fieldName].vocab =
                    this.appState.getSpecKeysFromFieldMatcher(
                        spec[fieldName], this.modelName);
              }
            });
            return html`
                <lit-generator-controls .spec=${spec} .name=${genName}>
                </lit-generator-controls>`;
          })}
        </div>
    `;
    // clang-format on
  }

  renderEntry(mutableInput: Input, key: string, editable: boolean) {
    const value = mutableInput[key];
    const isCategorical =
        this.groupService.categoricalFeatureNames.includes(key);
    const handleInputChange = (e: Event) => {
      // tslint:disable-next-line:no-any
      mutableInput[key] = (e as any).target.value;
    };

    // For categorical outputs, render a dropdown.
    const renderCategoricalInput = () => {
      const catVals = this.groupService.categoricalFeatures[key];
      // clang-format off
      return html`
        <select class="dropdown" @change=${handleInputChange}>
          ${catVals.map(val => {
            return html`
              <option value="${val}" ?selected=${val === value}>
                ${val}
              </option>`;
          })}
        </select>`;
      // clang-format on
    };

    // For non-categorical outputs, render an editable textfield.
    // TODO(lit-dev): Consolidate this logic with the datapoint editor,
    // ideally as part of b/172597999.
    const renderFreeformInput = () => {
      const fieldSpec = this.appState.currentDatasetSpec[key];
      const nonEditableSpecs: LitName[] = ['EdgeLabels', 'SpanLabels'];
      editable =  editable && !isLitSubtype(fieldSpec, nonEditableSpecs);
      const formattedVal = formatForDisplay(value, fieldSpec);
      return editable ? html`
      <input type="text" class="input-box" @input=${handleInputChange}
      .value="${formattedVal}" />` : html`<div>${formattedVal}</div>`;
    };

    // Note the "." before "value" in the template below - this is to ensure
    // the value gets set by the template.
    // clang-format off
    const classes = classMap({'entry': true, 'text': !isCategorical});
    return html`
      <div class=${classes}>
          ${isCategorical ? renderCategoricalInput() : renderFreeformInput()}
      </div>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generator-module': GeneratorModule;
  }
}
