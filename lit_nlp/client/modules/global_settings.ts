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

/**
 * LIT App global settings menu
 */

// tslint:disable:no-new-decorators
// mwc-radio import placeholder - DO NOT REMOVE
// mwc-formfield import placeholder - DO NOT REMOVE
// mwc-textfield import placeholder - DO NOT REMOVE
import '@material/mwc-formfield';
import '@material/mwc-radio';
import '@material/mwc-textfield';
import '../elements/checkbox';

import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {action, computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {ApiService, AppState, SettingsService} from '../services/services';

import {styles} from './global_settings.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * The global settings menu
 */
@customElement('lit-global-settings')
export class GlobalSettingsComponent extends MobxLitElement {
  @property({type: Boolean}) isOpen = false;
  @property({type: Object}) close = () => {};

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly apiService = app.getService(ApiService);
  private readonly appState = app.getService(AppState);
  private readonly settingsService = app.getService(SettingsService);

  @observable private selectedDataset: string = '';
  @observable private selectedLayout: string = '';
  @observable private readonly modelCheckboxValues = new Map<string, boolean>();
  @observable private pathForDatapoints: string = '';
  @observable private datapointsStatus: string = '';

  @observable private hoveredModel: string = '';

  @computed
  get selectedModels() {
    const modelEntries = [...this.modelCheckboxValues.entries()];
    return modelEntries.filter(([modelName, isSelected]) => isSelected)
        .map(([modelName, isSelected]) => modelName);
  }


  // tslint:disable-next-line:no-any
  updated(changedProperties: Map<string, any>) {
    // Because this component is always rendered, it just changes from open to
    // closed state, we want to initialize it's local state and rerender
    // whenever it changes from closed to open.
    if (changedProperties.has('isOpen') && this.isOpen) {
      this.initializeLocalState();
      this.requestUpdate();
    }
  }

  @computed
  get datapointButtonsDisabled() {
    return this.pathForDatapoints === '';
  }

  @action
  initializeLocalState() {
    this.modelCheckboxValues.clear();
    Object.keys(this.appState.metadata.models).forEach(modelName => {
      this.modelCheckboxValues.set(modelName, false);
    });
    this.appState.currentModels.forEach(modelName => {
      this.modelCheckboxValues.set(modelName, true);
    });

    this.selectedDataset = this.appState.currentDataset;
    this.selectedLayout = this.appState.layoutName;
  }

  @action
  submitSettings() {
    const models = this.selectedModels;
    const dataset = this.selectedDataset;
    const layoutName = this.selectedLayout;

    this.settingsService.updateSettings({
      models,
      dataset,
      layoutName,
    });
  }

  render() {
    const hiddenClassMap = classMap({hide: !this.isOpen});

    // clang-format off
    return html`
      <div id="global-settings-holder">
        <div id="overlay" class=${hiddenClassMap}></div>
        <div id="global-settings" class=${hiddenClassMap}>
          <div id="table-holder">
            ${this.renderModelsConfig()}
            ${this.renderDatasetConfig()}
            ${this.renderLayoutConfig()}
            ${this.appState.metadata.demoMode ?
              null : this.renderDatapointsConfig()}
          </div>
          ${this.renderButtons()}
        </div>
      </div>
    `;
    // clang-format on
  }

  renderButtons() {
    const cancel = () => {
      this.close();
    };

    const submit = () => {
      this.submitSettings();
      this.close();
    };

    const noModelsSelected = this.selectedModels.length === 0;
    const dataset = this.selectedDataset;
    const datasetValid = this.settingsService.isDatasetValidForModels(
        dataset, this.selectedModels);
    const submitDisabled = noModelsSelected || !datasetValid;

    let errorMessage = '';
    if (noModelsSelected) {
      errorMessage = 'No models selected...';
    } else if (!datasetValid) {
      errorMessage = 'Selected models incompatible with selected dataset...';
    }

    return html`
      <div id="buttons-container">
        <div id="error-message">${errorMessage}</div>
        <div id="buttons">
          <button
            ?disabled=${submitDisabled}
            @click=${submit}>Submit
          </button>
          <button @click=${cancel}>Cancel</button>
        </div>
      </div>
    `;
  }

  renderModelsConfig() {
    const availableModels = [...this.modelCheckboxValues.keys()];

    const renderModelSelect = (modelName: string) => {
      const checked = this.modelCheckboxValues.get(modelName) === true;
      // tslint:disable-next-line:no-any
      const change = (e: any) => {
        this.modelCheckboxValues.set(modelName, e.target.checked);
      };
      const mouseEnter = () => {
        this.hoveredModel = modelName;
      };
      const mouseLeave = () => {
        this.hoveredModel = '';
      };

      return html`
        <div class="config-line">
          <mwc-formfield
            label=${modelName}
            @mouseenter=${mouseEnter}
            @mouseleave=${mouseLeave}
          >
            <lit-checkbox ?checked=${checked} @change=${change}></lit-checkbox>
          </mwc-formfield>
        </div>
      `;
    };

    return html`
      <div id="models-config">
        <div class="config-title">Models</div>
        <div class="config-models-list">
          ${availableModels.map(name => renderModelSelect(name))}
        </div>
      </div>
    `;
  }

  renderDatasetConfig() {
    const allDatasets = Object.keys(this.appState.metadata.datasets);

    // If the currently-selected dataset is now invalid given the selected
    // models then search for a valid dataset to be pre-selected.
    if (!this.settingsService.isDatasetValidForModels(
            this.selectedDataset, this.selectedModels)) {
      for (let i = 0; i < allDatasets.length; i++) {
        const dataset = allDatasets[i];
        if (this.settingsService.isDatasetValidForModels(
                dataset, this.selectedModels)) {
          this.selectedDataset = dataset;
          break;
        }
      }
    }

    const renderDatasetSelect = (dataset: string) => {
      const handleDatasetChange = () => {
        this.selectedDataset = dataset;
      };

      const checked = this.selectedDataset === dataset;
      const modelsToCheck = [
        ...this.selectedModels,
        ...(this.hoveredModel ? [this.hoveredModel] : []),
      ];

      const disabled =
          !this.settingsService.isDatasetValidForModels(dataset, modelsToCheck);

      const radioStyle = {
        opacity: disabled ? '0.2' : '1',
      };
      return html`
        <div class="config-line" style=${styleMap(radioStyle)}>
          <mwc-formfield label=${dataset}>
            <mwc-radio
              name="dataset"
              class="select-dataset"
              data-dataset=${dataset}
              ?checked=${checked}
              ?disabled=${disabled}
              @change=${handleDatasetChange}
            >
            </mwc-radio>
          </mwc-formfield>
        </div>
      `;
    };

    return html`
      <div id="models-config">
        <div class="config-title">Dataset</div>
        <div class="config-datasets-list">
          ${allDatasets.map(name => renderDatasetSelect(name))}
        </div>
      </div>
    `;
  }

  renderDatapointsConfig() {
    const updatePath = (e: Event) => {
      const input = e.target! as HTMLInputElement;
      this.pathForDatapoints = input.value;
    };
    const save = async () => {
      const newDatapoints = this.appState.currentInputData.filter(
          input => input.meta['added'] === 1);
      if (newDatapoints.length === 0) {
        this.datapointsStatus = 'No new datapoints to save';
        return;
      }
      const newPath = await this.apiService.saveDatapoints(
          newDatapoints, this.appState.currentDataset, this.pathForDatapoints);
      for (const datapoint of newDatapoints) {
        datapoint.meta['added'] = 0;
      }
      this.datapointsStatus =
          `Saved ${newDatapoints.length} datapoints at ${newPath}`;
    };

    const load = async () => {
      const dataset = this.appState.currentDataset;
      const models = this.appState.currentModels;
      const datapoints =
          await this.apiService.loadDatapoints(dataset, this.pathForDatapoints);
      if (datapoints == null || datapoints.length === 0) {
        this.datapointsStatus =
            `No persisted datapoints found in ${this.pathForDatapoints}`;
        return;
      }
      for (const datapoint of datapoints) {
        datapoint.meta['added'] = 0;
      }
      // Update input data for new datapoints.
      // TODO(lit-dev): consolidate this update logic in appState.
      datapoints.forEach(entry => {
        this.appState.currentInputDataById.set(entry.id, entry);
      });
      this.datapointsStatus = `Loaded ${datapoints.length} datapoints from ${
          this.pathForDatapoints}`;
    };

    return html`
      <div id="datapoints-config">
        <div class="config-title">Generated Datapoints</div>
        <label for="path">Path for new datapoints:</label><br>
        <input type="text" name="path" value=${this.pathForDatapoints} @input=${
        updatePath}><br>
        <div>
          <button class="first-button"
            ?disabled=${this.datapointButtonsDisabled}
            @click=${save}
          >Save new datapoints
          </button>
          <button
            ?disabled=${this.datapointButtonsDisabled}
            @click=${load}
          >Load new datapoints
          </button>
        </div>
        <div>${this.datapointsStatus}</div>
        </div>
      </div>
    `;
  }
  renderLayoutConfig() {
    const layouts = Object.keys(this.appState.layouts);
    const renderLayoutOption = (name: string) => html`
    <div class="config-line">
      <mwc-formfield label=${name}>
        <mwc-radio
          name="layouts"
          ?checked=${this.selectedLayout === name}
          @change=${() => this.selectedLayout = name}
        >
        </mwc-radio>
      </mwc-formfield>
    </div>
  `;

    return html`
      <div id="layout-config">
        <div class="config-title">Layout</div>
        <div>
          ${layouts.map(name => renderLayoutOption(name))}
        </div>
      </div>
    `;  }
}


declare global {
  interface HTMLElementTagNameMap {
    'lit-global-settings': GlobalSettingsComponent;
  }
}
