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

import '../elements/interpreter_controls';
import '@material/mwc-icon';

import {MobxLitElement} from '@adobe/lit-mobx';
import {css, html, TemplateResult} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {TableData, TableEntry} from '../elements/table';
import {EdgeLabels, FieldMatcher, LitTypeTypesList, SpanLabels} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, formatForDisplay, IndexedInput, Input, ModelInfoMap, Spec} from '../lib/types';
import {cloneSpec, flatten, isLitSubtype} from '../lib/utils';
import {GroupService} from '../services/group_service';
import {SelectionService, SliceService} from '../services/services';

import {styles} from './generator_module.css';

/**
 * Custom element for in-table add/remove controls.
 * We use a custom element here so we can encapsulate styles.
 */
@customElement('generated-row-controls')
export class GeneratedRowControls extends MobxLitElement {
  static override get styles() {
    return [
      sharedStyles, styles, css`
            :host {
              display: flex;
              flex-direction: row;
              color: #1a73e8;
            }
            :host > * {
              margin-right: 4px;
            }
            `
    ];
  }

  override render() {
    const addPoint = () => {
      const event = new CustomEvent('add-point');
      this.dispatchEvent(event);
    };

    const removePoint = () => {
      const event = new CustomEvent('remove-point');
      this.dispatchEvent(event);
    };

    // clang-format off
    return html`
      <mwc-icon class="icon-button outlined" @click=${addPoint}>
        add_box
      </mwc-icon>
      <mwc-icon class="icon-button outlined" @click=${removePoint}>
        delete
      </mwc-icon>
    `;
    // clang-format on
  }
}

/**
 * A LIT module that allows the user to generate new examples.
 */
@customElement('generator-module')
export class GeneratorModule extends LitModule {
  static override title = 'Datapoint Generator';
  static override referenceURL =
      'https://github.com/PAIR-code/lit/wiki/components.md#generators';
  static override numCols = 10;

  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <generator-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </generator-module>`;

  static override duplicateForModelComparison = false;
  private readonly groupService = app.getService(GroupService);
  private readonly sliceService = app.getService(SliceService);

  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable editedData: Input = {};
  @observable isGenerating = false;

  @observable generated: IndexedInput[][] = [];
  @observable appliedGenerator: string|null = null;
  @observable sliceName: string = '';

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
  get globalParams() {
    return {
      'model_name': this.modelName,
      'dataset_name': this.datasetName,
    };
  }

  @computed
  get totalNumGenerated() {
    return this.generated.reduce((a, b) => a + b.length, 0);
  }

  override firstUpdated() {
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

  private resetEditedData() {
    this.generated = [];
    this.appliedGenerator = null;
    this.sliceName = '';
  }

  private handleGeneratorClick(generator: string, config?: CallConfig) {
    if (!this.isGenerating) {
      this.resetEditedData();
      this.generate(generator, this.modelName, config);
    }
  }

  private makeAutoSliceName(generator: string, config?: CallConfig) {
    const segments: string[] = [generator];
    if (config != null) {
      for (const key of Object.keys(config)) {
        // Skip these since they don't come from the actual controls form.
        if (this.globalParams.hasOwnProperty(key)) continue;
        segments.push(`${key}=${config[key]}`);
      }
    }
    return segments.join(':');
  }

  private async generate(
      generator: string, modelName: string, config?: CallConfig) {
    this.isGenerating = true;
    this.appliedGenerator = generator;
    const sourceExamples = this.selectionService.selectedInputData;
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
      this.isGenerating = false;
      this.sliceName = this.makeAutoSliceName(generator, config);
    } catch {
      this.isGenerating = false;
    }
  }

  private async createNewDatapoints(data: IndexedInput[][]) {
    const newExamples = flatten(data);
    this.appState.commitNewDatapoints(newExamples);
    const newIds = newExamples.map(d => d.id);
    if (newIds.length === 0) return;

    if (this.sliceName !== '') {
      this.sliceService.addOrAppendToSlice(this.sliceName, newIds);
    }

    const parentIds =
        new Set<string>(newExamples.map(ex => ex.meta['parentId']!));

    // Select parents and children, and set primary to the first child.
    this.selectionService.selectIds([...parentIds, ...newIds], this);
    this.selectionService.setPrimarySelection(newIds[0], this);

    // If in comparison mode, set reference selection to the parent point
    // for direct comparison.
    if (this.appState.compareExamplesEnabled) {
      const referenceSelectionService =
          app.getService(SelectionService, 'pinned');
      referenceSelectionService.selectIds([...parentIds, ...newIds], this);
      // parentIds[0] is not necessarily the parent of newIds[0], if
      // generated[0] is [].
      const parentId = newExamples[0].meta['parentId']!;
      referenceSelectionService.setPrimarySelection(parentId, this);
    }
  }

  override renderImpl() {
    return html`
      <div class="module-container">
        <div class="module-content generator-module-content">
          ${this.renderGeneratorButtons()}
          ${this.renderGenerated()}
        </div>
        <div class="module-footer">
          <p class="module-status">${this.getStatus()}</p>
          <p class="module-status">${this.getSliceStatus()}</p>
          ${this.renderOverallControls()}
        </div>
      </div>
    `;
  }

  getSliceStatus(): TemplateResult|null {
    const sliceExists = this.sliceService.namedSlices.has(this.sliceName);
    if (!sliceExists) return null;

    const existingSliceSize =
        this.sliceService.getSliceByName(this.sliceName)!.length;
    const s = existingSliceSize === 1 ? '' : 's';
    // clang-format off
    return html`
       Slice exists (${existingSliceSize} point${s});
       will append ${this.totalNumGenerated} more.
    `;
    // clang-format on
  }

  /**
   * Determine module's status as a string to display in the footer.
   */
  getStatus(): string|TemplateResult {
    if (this.isGenerating) {
      return 'Generating...';
    }

    if (this.appliedGenerator) {
      const s = this.totalNumGenerated === 1 ? '' : 's';
      return `
        ${this.appliedGenerator}: generated ${this.totalNumGenerated}
        counterfactual${s} from ${this.generated.length} inputs.
      `;
    }

    const generatorsInfo = this.appState.metadata.generators;
    const generators = Object.keys(generatorsInfo);
    if (!generators.length) {
      return 'No generator components available.';
    }

    const data = this.selectionService.selectedInputData;
    const selectAll = () => {
      this.selectionService.selectAll();
    };
    if (data.length <= 0) {
      return html`
          No examples selected.
          <span class='select-all' @click=${selectAll}>
            Select entire dataset?
          </span>`;
    }

    const s = data.length === 1 ? '' : 's';
    return `
      Generate counterfactuals from current selection
      (${data.length} datapoint${s}).
    `;
  }

  renderInterstitial() {
    // clang-format off
    return html`
      <div class="interstitial">
        <img src="static/interstitial-select.png" />
        <p>
          <strong>Counterfactual Generators</strong>
          Create new datapoints derived from the current selection.
        </p>
      </div>`;
    // clang-format on
  }

  renderEmptyNotice() {
    // clang-format off
    return html`
      <div class="interstitial">
        <p>No examples generated.</p>
      </div>`;
    // clang-format on
  }

  renderGenerated() {
    const rows: TableData[] = this.createEntries();

    if (!this.appliedGenerator) {
      return this.renderInterstitial();
    }

    if (rows.length <= 0) {
      if (this.isGenerating) {
        return null;
      } else {
        return this.renderEmptyNotice();
      }
    }

    // clang-format off
    return html`
      <div class="results-holder">
        <lit-data-table class="table"
            .columnNames=${Object.keys(rows[0])}
            .data=${rows}
            exportEnabled
        ></lit-data-table>
      </div>
    `;
    // clang-format on
  }

  /**
   * Render the generated counterfactuals themselves.
   */
  createEntries() {
    const rows: TableData[] = [];
    for (let parentIndex = 0; parentIndex < this.generated.length;
         parentIndex++) {
      const generatedList = this.generated[parentIndex];
      for (let generatedIndex = 0; generatedIndex < generatedList.length;
           generatedIndex++) {
        const generated = generatedList[generatedIndex];
        const addPoint = async () => {
          this.generated[parentIndex].splice(generatedIndex, 1);
          await this.createNewDatapoints([[generated]]);
        };
        const removePoint = () => {
          this.generated[parentIndex].splice(generatedIndex, 1);
        };
        const fieldNames = Object.keys(generated.data);

        // render values for each datapoint.
        const row: {[key: string]: TableEntry} = {};
        for (const key of fieldNames) {
          const editable =
              !this.appState.currentModelRequiredInputSpecKeys.includes(key);
          row[key] = editable ? this.renderEntry(generated.data, key) :
                                generated.data[key];
        }
        row['Add/Remove'] = html`<generated-row-controls
                                @add-point=${addPoint}
                                @remove-point=${removePoint} />`;
        rows.push(row);
      }
    }
    return rows;
  }

  renderOverallControls() {
    const onAddAll = async () => {
      await this.createNewDatapoints(this.generated);
      this.resetEditedData();
    };

    const onClickCompare = async () => {
      this.appState.compareExamplesEnabled = true;
      // this.createNewDatapoints() will set reference selection if
      // comparison mode is enabled.
      await onAddAll();
    };

    const setSliceName = (e: Event) => {
      // tslint:disable-next-line:no-any
      this.sliceName = (e as any).target.value;
    };

    const controlsDisabled = this.totalNumGenerated <= 0;

    // clang-format off
    return html`
      <div class="overall-controls">
        <label for="slice-name">Slice name:</label>
        <input type="text" class="slice-name-input" name="slice-name"
         .value=${this.sliceName} @input=${setSliceName}
         placeholder="Name for generated set">
        <button class="hairline-button" ?disabled=${controlsDisabled}
           @click=${onAddAll}>
           Add all
        </button>
        <button class='hairline-button'
          @click=${onClickCompare} ?disabled=${controlsDisabled}>
          Add and compare
        </button>
        <button class="hairline-button" ?disabled=${controlsDisabled}
           @click=${this.resetEditedData}>
           Clear
        </button>
      </div>`;
    // clang-format on
  }

  renderGeneratorButtons() {
    const generatorsInfo = this.appState.metadata.generators;
    const generators = Object.keys(generatorsInfo);

    // Add event listener for generation events.
    const onGenClick = (event: CustomEvent) => {
      // tslint:disable-next-line:no-any
      const generatorParams: {[setting: string]: string} =
          event.detail.settings;
      // tslint:disable-next-line:no-any
      const generatorName = event.detail.name;

      // Add user-specified parameters from the applied generator.
      const allParams = Object.assign({}, this.globalParams, generatorParams);
      this.handleGeneratorClick(generatorName, allParams);
    };

    // clang-format off
    return html`
        <div class="generators-panel">
          ${generators.map(genName => {
      const spec = generatorsInfo[genName].configSpec;
      const clonedSpec = cloneSpec(spec);
      const description = generatorsInfo[genName].description;
      for (const fieldSpec of Object.values(clonedSpec)) {
        // If the generator uses a field matcher, then get the matching
        // field names from the specified spec and use them as the vocab.
        if (fieldSpec instanceof FieldMatcher) {
          fieldSpec.vocab = this.appState.getSpecKeysFromFieldMatcher(
              fieldSpec, this.modelName);
        }
      }
      return html`
                <lit-interpreter-controls
                  .spec=${clonedSpec}
                  .name=${genName}
                  .description=${description || ''}
                  @interpreter-click=${onGenClick}>
                </lit-interpreter-controls>`;
    })}
        </div>
    `;
    // clang-format on
  }

  renderEntry(mutableInput: Input, key: string) {
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
      return {
        template: html`
          <select class="dropdown" @change=${handleInputChange}>
            ${catVals.map(val => {
              return html`
                <option value="${val}" ?selected=${val === value}>
                  ${val}
                </option>`;
            })}
          </select>`,
        value
      };
      // clang-format on
    };

    // For non-categorical outputs, render an editable textfield.
    // TODO(lit-dev): Consolidate this logic with the datapoint editor,
    // ideally as part of b/172597999.
    const renderFreeformInput = () => {
      const fieldSpec = this.appState.currentDatasetSpec[key];
      const nonEditableSpecs: LitTypeTypesList = [EdgeLabels, SpanLabels];
      const formattedVal = formatForDisplay(value, fieldSpec).toString();

      if (isLitSubtype(fieldSpec, nonEditableSpecs)) {
        return formattedVal;
      }

      return {
        template: html`<input type="text" class="input-box"
            @input=${handleInputChange}
            .value="${formattedVal}" />`,
        value: formattedVal,
      };
    };

    return isCategorical ? renderCategoricalInput() : renderFreeformInput();
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generator-module': GeneratorModule;
  }
}
