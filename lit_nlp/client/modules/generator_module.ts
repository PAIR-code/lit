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
import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable, reaction} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {CallConfig, IndexedInput, Input, ModelsMap, Spec} from '../lib/types';
import {findSpecKeys, handleEnterKey} from '../lib/utils';
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
  static numCols = 12;

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
  // Source examples stores the original examples used to create this.generated,
  // and should be maintained as a parallel list so that the correct parent
  // pointers can be set when the examples are added.
  // TODO(lit-dev): consider setting parent pointers immediately and storing
  // this.generated as IndexedInput[][] so we don't need to track two lists.
  @observable sourceExamples: IndexedInput[] = [];
  @observable generated: Input[][] = [];
  @observable appliedGenerator: string|null = null;
  @observable datapointEdited: boolean = false;
  @observable substitutions = 'great -> terrible';
  // When embedding indices is computed, this is the index of the selection.
  @observable embeddingSelection = 0;

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
  get embeddingFields() {
    const modelName = this.appState.currentModels[0];
    const modelSpec = this.appState.currentModelSpecs[modelName].spec;
    return findSpecKeys(modelSpec.output, 'Embeddings');
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
  }

  private resetEditedData() {
    this.sourceExamples = [];
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
    this.sourceExamples = this.selectionService.selectedOrAllInputData;
    const generated = await this.apiService.getGenerated(
        this.sourceExamples, modelName, this.appState.currentDataset, generator,
        config);
    this.generated = generated;
    this.appliedGenerator = generator;
    this.isGenerating = false;
  }

  private async createNewDatapoints(
      data: Input[][], parentIds: string[], source: string) {
    const newExamples =
        await this.appState.createNewDatapoints(data, parentIds, source);
    const newIds = newExamples.map(d => d.id);
    if (newIds.length === 0) return;

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
                'counterfactual' : 'counterfactuals'}.
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
    const data = this.sourceExamples;
    return this.generated.map((generatedList, parentIndex) => {
      return generatedList.map((generated, generatedIndex) => {
        const addPoint = async () => {
          const parentId = data[parentIndex].id;
          this.generated[parentIndex].splice(generatedIndex, 1);
          await this.createNewDatapoints(
              [[generated]], [parentId], this.appliedGenerator!);
        };
        const removePoint = () => {
          this.generated[parentIndex].splice(generatedIndex, 1);
        };
        const keys = Object.keys(generated);

        // render values for each datapoint.
        const editable = false;
        // clang-format off
        return html`
          <div class='row'>
            ${keys.map((key) => this.renderEntry(key, generated[key], editable))}
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
      const parentIds = this.sourceExamples.map((datapoint) => datapoint.id);
      await this.createNewDatapoints(
          this.generated, parentIds, this.appliedGenerator!);
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
    const generators = this.appState.metadata.generators;
    const text = this.appState.metadata.generators.length > 0 ?
        `Generate counterfactuals for current selection (${
            data.length} datapoint${data.length === 1 ? '' : `s`}):` :
        'No generators provided by the server.';
    // TODO(lit-team): Add configurable controls for each generator.
    const updateSubstitutions = (e: Event) => {
      const input = e.target! as HTMLInputElement;
      this.substitutions = input.value;
    };

    const renderEmbeddingSelection = (embeddingFieldNames: string[]) => {
      const updateEmbeddingSelection = (e: Event) => {
        const select = (e.target as HTMLSelectElement);
        this.embeddingSelection = select?.selectedIndex || 0;
      };
      const options = embeddingFieldNames.map((option, optionIndex) => {
        return html`
          <option value=${optionIndex}>${option}</option>
        `;
      });
      const defaultValue = embeddingFieldNames[0];

      return html`
      <span class="dropdown-container">
        <label class="dropdown-label">Embedding</label>
        <select class="dropdown" @change=${updateEmbeddingSelection} .value=${
          defaultValue}>
          ${options}
        </select>
      </span>
    `;
    };

    const genericGenerator = (generator: string) => {
      const onClick = () => {
        this.handleGeneratorClick(generator);
      };
      // clang-format off
      return html`
        <div class='generator-control'>
          <button @click=${onClick}>${generator}</button>
        </div>`;
      // clang-format on
    };
    const similaritySearcher = (generator: string) => {
      const onClick = () => {
        this.handleGeneratorClick(generator, {
          'model_name': this.modelName,
          'dataset_name': this.datasetName,
          'field_name': this.embeddingFields[this.embeddingSelection],
        });
      };
      return html`
        <div>
          <button @click=${onClick}>${generator}</button>
          ${renderEmbeddingSelection(this.embeddingFields)}
        </div>`;
    };
    const wordReplacer = (generator: string) => {
      const onClick = () => {
        this.handleGeneratorClick(generator, {'subs': this.substitutions});
      };
      // clang-format off
      return html`
        <div class='generator-control'>
          <button @click=${onClick}>${generator}</button>
          <label class="input-label">Substitutions: </label>
          <input type="text" style=${styleMap({width: "40%"})}
            @input=${updateSubstitutions}
            .value="${this.substitutions}" />
        </div>`;
      // clang-format on
    };
    // clang-format off
    return html`
        <div id="generators">
          <div>${text}</div>
          ${generators.map(generator => {
      if (generator === 'similarity_searcher') {
        return similaritySearcher(generator);
      }
      if (generator === 'word_replacer') {
        return wordReplacer(generator);
      } else {
        return genericGenerator(generator);
      }
    })}
        </div>
    `;
    // clang-format on
  }

  renderEntry(key: string, value: string, editable: boolean) {
    const isCategorical =
        this.groupService.categoricalFeatureNames.includes(key);
    const handleInputChange = (e: Event) => {
      this.datapointEdited = true;
      // tslint:disable-next-line:no-any
      this.editedData[key] = (e as any).target.value;
    };

    // For categorical outputs, render a dropdown.
    const renderCategoricalInput = () => {
      const catVals = this.groupService.categoricalFeatures[key];
      // Note that the first option is blank (so that the dropdown is blank when
      // no point is selected), and disabled (so that datapoints can only have
      // valid values).
      return html`
      <select class="dropdown"
        @change=${handleInputChange}>
        <option value="" selected></option>
        ${catVals.map(val => {
        return html`
            <option
              value="${val}"
              ?selected=${val === value}
              >
              ${val}
            </option>`;
      })}
      </select>`;
    };

    // For non-categorical outputs, render an editable textfield.
    const renderFreeformInput = () => {
      return editable ? html`
      <input type="text" class="input-box" @input=${handleInputChange}
      .value="${value}" />` :
                        html`<div>${value}</div>`;
    };

    const onKeyUp = (e: KeyboardEvent) => {
      handleEnterKey(e, () => {
        this.shadowRoot!.getElementById('make')!.click();
      });
    };

    // Note the "." before "value" in the template below - this is to ensure
    // the value gets set by the template.
    // clang-format off
    const classes = classMap({'entry': true, 'text': !isCategorical});
    return html`
      <div class=${classes} @keyup=${(e: KeyboardEvent) => {onKeyUp(e);}}>
          ${isCategorical ? renderCategoricalInput() : renderFreeformInput()}
      </div>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'generator-module': GeneratorModule;
  }
}
