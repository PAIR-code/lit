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
import {html, TemplateResult} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {action, computed, observable} from 'mobx';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {StringLitType} from '../lib/lit_types';
import {type CallConfig, datasetDisplayName, LitTabGroupLayout, NONE_DS_DICT_KEY, Spec} from '../lib/types';
import {getTemplateStringFromMarkdown, validateCallConfig} from '../lib/utils';
import {LitInputField} from '../elements/lit_input_field';
import {resolveModuleConfig} from '../services/modules_service';
import {ApiService, AppState, SettingsService} from '../services/services';

import {app} from './app';
import {styles} from './global_settings.css';

type EventHandler = (e: Event) => void;

/**
 * Names of available settings tabs.
 */
export type TabName = 'Models'|'Dataset'|'Layout';
const MODEL_DESC = 'Select models to explore in LIT.';
const DATASET_DESC =
    'Select a compatible dataset to use with the selected models.';
const LAYOUT_DESC = 'Use a preset layout to optimize your workflow';

const SELECTED_TXT = 'Selected';
const COMPATIBLE_TXT = 'Compatible';
const INCOMPATIBLE_TXT = 'Incompatible';

const NEW_NAME_FIELD = 'new_name';
const LOAD_DISABLED_TXT = 'Provide a value for "new_name" to load';

function initializeCallConfig(spec: Spec): CallConfig {
  const config: CallConfig = {};
  for (const [key, litType] of Object.entries(spec)) {
    if (litType.default != null) {config[key] = litType.default;}
  }
  return config;
}

/**
 * The global settings menu
 */
@customElement('lit-global-settings')
export class GlobalSettingsComponent extends MobxLitElement {
  @property({type: Boolean}) isOpen = false;

  static override get styles() {
    return [sharedStyles, styles];
  }
  private readonly apiService = app.getService(ApiService);
  private readonly appState = app.getService(AppState);
  private readonly settingsService = app.getService(SettingsService);

  @observable private selectedDataset = '';
  @observable private selectedLayout = '';
  @observable private readonly modelCheckboxValues = new Map<string, boolean>();
  @observable selectedTab: TabName = 'Models';
  @observable private status?: string;

  // TODO(b/207137261): Determine if datapointsStatus, modelStatus,
  // pathForDatapoints, and pathForModel are still necessary/how to use convey
  // this information to the user following the load-from-init_spec refactor.
  // @observable private pathForDatapoints = '';
  // @observable private datapointsStatus = '';
  // @observable private pathForModel = '';
  // @observable private modelStatus = '';

  // tslint:disable:no-inferrable-new-expression
  @observable private readonly openModelKeys: Set<string> = new Set();
  @observable private readonly openDatasetKeys: Set<string> = new Set();
  @observable private readonly openLayoutKeys: Set<string> = new Set();

  @observable private datasetToLoad?: string;
  @observable private modelToLoad?: string;
  @observable private loadingCallConfig: CallConfig = {};
  @observable private missingCallConfigFields: string[] = [];

  @computed get loadableDatasets(): string[] {
    const {datasets} = this.appState.metadata.initSpecs;
    const loadable = Object.entries(datasets)
        .filter(([unused, spec]) => spec != null)
        .map(([name, unused]) => name);
    return loadable;
  }

  @computed get loadableModels(): string[] {
    const {models} = this.appState.metadata.initSpecs;
    const loadable = Object.entries(models)
        .filter(([unused, spec]) => spec != null)
        .map(([name, unused]) => name);
    return loadable;
  }

  @computed get selectedModels() {
    const modelEntries = [...this.modelCheckboxValues.entries()];
    return modelEntries.filter(([, isSelected]) => isSelected)
        .map(([modelName,]) => modelName);
  }

  // TODO(b/207137261): Determine where and how dataset saving happens after the
  // load from init_spec() refactor.
  // @computed get saveDatapointButtonDisabled() {
  //   return this.pathForDatapoints === '' || this.newDatapoints.length === 0 ||
  //       this.appState.currentDataset !== this.selectedDataset;
  // }
  //
  // @computed get newDatapoints() {
  //  return this.appState.currentInputData.filter(input => input.meta.added);
  // }

  /**
   * Open the settings menu.
   */
  open() {
    // Initialize local state (selected models, data, etc.) from app state.
    this.initializeLocalState();
    this.requestUpdate();
    this.isOpen = true;
    this.resetLoadingCallConfig();
  }

  private resetLoadingCallConfig() {
    let spec;

    if (this.selectedTab === 'Dataset') {
      const name = this.datasetToLoad || this.loadableDatasets[0];
      spec = this.appState.metadata.initSpecs.datasets[name];
    } else if (this.selectedTab === 'Models') {
      const name = this.modelToLoad || this.loadableModels[0];
      spec = this.appState.metadata.initSpecs.models[name];
    }

    if (spec != null) {
      this.loadingCallConfig = initializeCallConfig(spec);
    }
  }

  /**
   * Close the settings menu.
   */
  close() {
    this.isOpen = false;
  }

  @action
  initializeLocalState() {
    this.modelCheckboxValues.clear();
    Object.keys(this.appState.metadata.models)
        .filter(modelName => !modelName.startsWith('_'))  // hidden models
        .forEach(modelName => {
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

  override render() {
    const hiddenClassMap = classMap({'hide': !this.isOpen});
    // clang-format off
    return html`
      <div id="global-settings-holder">
        <div id="overlay" class=${hiddenClassMap}
         @click=${() => { this.close(); }}></div>
        <div id="global-settings" class=${hiddenClassMap}>
        <div id="title-bar">
          Configure LIT

          <lit-tooltip content="Go to reference">
            <a slot="tooltip-anchor" target='_blank'
              href='https://pair-code.github.io/lit/documentation/ui_guide.md#global-settings'>
            <mwc-icon class="icon-button large-icon" id="configure-lit-icon">
              open_in_new
            </mwc-icon>
            </a>
          </lit-tooltip>
        </div>
        <div id="holder">
          <div id="sidebar">
            ${this.renderTabs()}
            ${this.renderLinks()}
          </div>
          <div id="main-panel">
            ${this.renderConfig()}
            ${this.renderBottomBar()}
          </div>
        </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  /** Render the control tabs. */
  private renderTabs() {
    const tabs: TabName[] = ['Models', 'Dataset', 'Layout'];
    const renderTab = (tab: TabName) => {
      const click = () => {
        this.selectedTab = tab;
        this.resetLoadingCallConfig();
      };
      const classes = classMap({
        'tab': true,
        'selected': this.selectedTab === tab
      });
      return html`<div class=${classes} @click=${click}>${tab}</div>`;
    };
    return html`
    <div id="tabs">
      ${tabs.map(tab => renderTab(tab))}
    </div>
    `;
  }

  /** Render the links at the bottom of the page. */
  private renderLinks() {
    const help =
        'https://pair-code.github.io/lit/tutorials';
    const github = 'https://github.com/PAIR-code/lit';
    return html`
    <div id="links">
      <a href=${github} class='link-out' target="_blank">
        Github
      </a>
      •
      <a href=${help} class='link-out' target="_blank">
        Help & Tutorials
      </a>
    </div>
    `;
  }

  /**
   * Render the bottom bar with the currently selected options, as well as
   * buttons.
   */
  private renderBottomBar() {
    const noModelsSelected = this.selectedModels.length === 0;
    const datasetValid = this.settingsService.isDatasetValidForModels(
        this.selectedDataset, this.selectedModels);

    const modelClasses = classMap({
      'info': true,
      'disabled': !this.selectedModels.length,
      'error': noModelsSelected
    });
    const modelsStr = this.selectedModels.length ?
        this.selectedModels.join(', ') :
        'No models selected';

    const datasetClasses = classMap({'info': true, 'error': !datasetValid});

    return html`
    <div id="bottombar">
      <div id="state">
        <div> selected model(s):
          <span class=${modelClasses}> ${modelsStr} </span>
        </div>
        <div> selected dataset:
          <span class=${datasetClasses}>
            ${datasetDisplayName(this.selectedDataset)}
          </span>
        </div>
      </div>
      <div class="status-area">
        ${this.status}
        ${this.status != null ? 'See error in lower left corner.' : ''}
      </div>
      <div> ${this.renderButtons(noModelsSelected, datasetValid)} </div>
    </div>
  `;
  }

  /** Render the submit and cancel buttons. */
  private renderButtons(noModelsSelected: boolean, datasetValid: boolean) {
    const cancel = () => {
      this.selectedTab = 'Models';
      this.close();
    };

    const submit = () => {
      this.submitSettings();
      cancel();
    };

    const submitDisabled = noModelsSelected || !datasetValid;
    return html`
      <div id="buttons-container">
        <div id="buttons">
          <button class="hairline-button" @click=${cancel}>Cancel</button>
          <button class="filled-button"
                  ?disabled=${submitDisabled}
                  @click=${submit}>
            Submit
          </button>
        </div>
      </div>
    `;
  }

  /** Render the config main page. */
  private renderConfig() {
    const tab = this.selectedTab;
    const configLayout = tab === 'Models' ?
        this.renderModelsConfig() :
        (tab === 'Dataset' ? this.renderDatasetConfig() :
                             this.renderLayoutConfig());
    return html`
    <div id="config">
      ${configLayout}
    </div>
    `;
  }

  /** Render the models page content. */
  renderModelsConfig() {
    const availableModels = [...this.modelCheckboxValues.keys()];
    const renderModelSelect = (name: string) => {
      const selected = this.modelCheckboxValues.get(name) === true;
      const disabled = false;
      const expanderOpen = this.openModelKeys.has(name);
      const onExpanderClick = () => {
        this.toggleInSet(this.openModelKeys, name);
      };
      // tslint:disable-next-line:no-any
      const change = (e: any) => {
        this.modelCheckboxValues.set(name, e.target.checked);
        // If the currently-selected dataset is now invalid given the selected
        // models then search for a valid dataset to be pre-selected.
        if (!this.settingsService.isDatasetValidForModels(
                this.selectedDataset, this.selectedModels)) {
          this.selectedDataset = NONE_DS_DICT_KEY;
          // Match the first compatible dataset.
          const allDatasets = Object.keys(this.appState.metadata.datasets);
          for (const dataset of allDatasets) {
            if (this.settingsService.isDatasetValidForModels(
                    dataset, this.selectedModels)) {
              this.selectedDataset = dataset;
              break;
            }
          }
        }
      };
      const selectorHtml = html`
          <mwc-formfield label=${name}>
            <lit-checkbox
              class='checkbox'
              ?checked=${selected}
              @change=${change}></lit-checkbox>
          </mwc-formfield>
      `;

      // Render the expanded info section, which holds the comparable
      // datasets and the description of the model.
      const allDatasets = Object.keys(this.appState.metadata.datasets);
      const ospec = this.appState.getModelSpec(name).output;
      // clang-format off
      const expandedInfoHtml = html`
        <div class='info-group-title'>
          Dataset Compatibility
        </div>
        ${allDatasets.map((datasetName: string) => {
          const compatible = this.settingsService.isDatasetValidForModels(
            datasetName, [name]);
          const error = !compatible && (this.selectedDataset === datasetName);
          const classes = classMap({
            'compatible': compatible,
            'error': error,
            'info-line': true
          });
          const icon = compatible ? 'check' : (error ? 'warning_amber' : 'clear');
          return html`
            <div class=${classes}>
              <span class='material-icon'>${icon}</span>
              ${datasetDisplayName(datasetName)}
            </div>`;
        })}
        <div class='info-group-title'>
          Output Fields
        </div>
        ${Object.keys(ospec).map((fieldName: string) => {
          return html`
            <div class='info-line'>
              ${fieldName} (${ospec[fieldName].name})
            </div>`;
        })}
      `;
      // clang-format on
      const description = this.appState.metadata.models[name].description;
      return this.renderLine(
          name, selectorHtml, selected, disabled, expanderOpen, onExpanderClick,
          false, expandedInfoHtml, description);
    };

    const configListHTML = availableModels.map(name => renderModelSelect(name));
    const buttonsHTML = html`${this.nextPrevButton('Dataset', true)}`;

    const loaderModel = this.modelToLoad || this.loadableModels[0];
    const loaderSpec = this.appState.metadata.initSpecs.models[loaderModel];

    const selectModel = (e: Event) => {
      this.modelToLoad = (e.target as HTMLSelectElement).value;
      const initSpec =
          this.appState.metadata.initSpecs.models[this.modelToLoad];
      if (initSpec != null) {
        this.loadingCallConfig = initializeCallConfig(initSpec);
      }
    };

    const loadModel = async () => {
      this.missingCallConfigFields = validateCallConfig(
        this.loadingCallConfig, loaderSpec);
      this.status = this.missingCallConfigFields.length ?
          "Missing required model initialization parameters." : undefined;

      if (this.missingCallConfigFields.length) return;

      this.status = undefined;
      let newInfo;

      try {
        newInfo = await this.apiService.createModel(
            loaderModel, this.loadingCallConfig);
      } catch {
        this.status = 'Unable to load model.';
      }

      if (newInfo == null) {return;}

      const [metadata, modelName] = newInfo;
      if (loaderSpec != null) {
        this.loadingCallConfig = initializeCallConfig(loaderSpec);
      }
      this.appState.metadata = metadata;
      this.appState.currentModels.push(modelName);
      this.initializeLocalState();
      // this.status = 'New model initialized and added auccessfully.';
    };

    const hideLoadingControls = this.appState.metadata.demoMode ||
                                loaderModel == null || loaderSpec == null;
    const loadingControls = hideLoadingControls ?
        html`` : this.renderLoader(
          'Model',
          this.loadableModels,
          loaderModel,
          loaderSpec,
          selectModel,
          loadModel
        );

    return this.renderConfigPage(
        'Models', MODEL_DESC, configListHTML, buttonsHTML, loadingControls);
  }

  /** Render the datasets page content. */
  renderDatasetConfig() {
    const allDatasets = Object.keys(this.appState.metadata.datasets);
    const renderDatasetSelect = (name: string) => {
      const displayName = datasetDisplayName(name);
      const handleDatasetChange = () => {
        // TODO(b/207137261): Update or remove path resetting behaivor after the
        // save datapoints functionality is resolved.
        // this.pathForDatapoints = '';
        // this.datapointsStatus = '';
        this.selectedDataset = name;
      };

      const selected = this.selectedDataset === name;
      const disabled = !this.settingsService.isDatasetValidForModels(
          name, this.selectedModels);

      const expanderOpen = this.openDatasetKeys.has(name);
      const onExpanderClick = () => {
          this.toggleInSet(this.openDatasetKeys, name);
        };
      // clang-format off
      const selectorHtml = html`
            <mwc-formfield label=${displayName}>
              <mwc-radio
                name="dataset"
                class="select-dataset"
                data-dataset=${displayName}
                ?checked=${selected}
                ?disabled=${disabled}
                @change=${handleDatasetChange}>
              </mwc-radio>
            </mwc-formfield>
      `;
      // clang-format on

      // Expanded info contains available datasets.
      const datasetInfo = this.appState.metadata.datasets[name];
      const allModels = [...this.modelCheckboxValues.keys()];
      // clang-format off
      const expandedInfoHtml = html`
        <div class='info-group-title'>
          ${datasetInfo.size} datapoint${datasetInfo.size !== 1 ? "s" : ""}
        </div>
        <div class='info-group-title'>
          Features
        </div>
        ${Object.keys(datasetInfo.spec).map((fieldName: string) => {
          return html`
            <div class='info-line'>
              ${fieldName} (${datasetInfo.spec[fieldName].name})
            </div>`;
        })}
        <div class='info-group-title'>
          Model Compatibility
        </div>
        ${allModels.map((modelName: string) => {
        const compatible = this.settingsService.isDatasetValidForModels(
          name, [modelName]);
        const error = !compatible && (this.selectedModels.includes(modelName));
        const classes = classMap({
          'compatible': compatible,
          'error': error,
          'info-line': true
        });
        const icon = compatible ? 'check' : (error ? 'warning_amber' : 'clear');
        return html`
        <div class=${classes}>
          <span class='material-icon'>${icon}</span>
          ${modelName}
        </div>`;
        })}
      `;
      // clang-format on
      const description = this.appState.metadata.datasets[name].description;
      return this.renderLine(
          name, selectorHtml, selected, disabled, expanderOpen, onExpanderClick,
          true, expandedInfoHtml, description);
    };
    const configListHTML = allDatasets.map(name => renderDatasetSelect(name));
    // clang-format off
    const buttonsHTML = html`
      ${this.nextPrevButton('Models', false)}
      ${this.nextPrevButton('Layout', true)}
    `;
    // clang-format on

    // TODO(b/207137261): Figure out where the save button should go after the
    // dataset loading refactor.
    // const save = async () => {
    //   const newPath = await this.apiService.saveDatapoints(
    //       this.newDatapoints, this.appState.currentDataset,
    //       this.pathForDatapoints);
    //   this.datapointsStatus =
    //       `Saved ${this.newDatapoints.length} datapoint` +
    //       `${this.newDatapoints.length === 1 ? '' : 's'} at ${newPath}`;
    // };
    // const load = async () => {
    //   const datapoints = await this.apiService.loadDatapoints(
    //       this.selectedDataset, this.pathForDatapoints);
    //   if (datapoints == null || datapoints.length === 0) {
    //     this.datapointsStatus =
    //         `No persisted datapoints found in ${this.pathForDatapoints}`;
    //     return;
    //   }
    //   // Update input data for new datapoints.
    //   this.appState.commitNewDatapoints(datapoints);
    //   this.datapointsStatus = `Loaded ${datapoints.length} ` +
    //       `datapoint${datapoints.length === 1 ? '' : 's'} from `+
    //       `${this.pathForDatapoints}`;
    // };
    const loaderDataset = this.datasetToLoad || this.loadableDatasets[0];
    const loaderSpec = this.appState.metadata.initSpecs.datasets[loaderDataset];

    const selectDataset = (e: Event) => {
      this.datasetToLoad = (e.target as HTMLSelectElement).value;
      const initSpec =
          this.appState.metadata.initSpecs.datasets[this.datasetToLoad];
      if (initSpec != null) {
        this.loadingCallConfig = initializeCallConfig(initSpec);
      }
    };

    const loadDataset = async () => {
      this.missingCallConfigFields = validateCallConfig(
        this.loadingCallConfig, loaderSpec);
      this.status = this.missingCallConfigFields.length ?
          "Missing required dataset initialization parameters." : undefined;

      if (this.missingCallConfigFields.length) return;

      let newInfo;

      try {
        newInfo = await this.apiService.createDataset(
          loaderDataset, this.loadingCallConfig);
      } catch {
        this.status = 'Unable to load dataset';
      }

      if (newInfo == null) {return;}
      const [metadata, datasetName] = newInfo;
      if (loaderSpec != null) {
        this.loadingCallConfig = initializeCallConfig(loaderSpec);
      }
      this.appState.metadata = metadata;
      this.selectedDataset = datasetName;
    };

    const hideLoadingControls = this.appState.metadata.demoMode ||
                                loaderDataset == null || loaderSpec == null;
    const loadingControls = hideLoadingControls ?
        html`` : this.renderLoader(
          'Dataset',
          this.loadableDatasets,
          loaderDataset,
          loaderSpec,
          selectDataset,
          loadDataset
        );

    return this.renderConfigPage(
        'Dataset', DATASET_DESC, configListHTML, buttonsHTML, loadingControls);
  }

  /**
   * Renders the UI for loading a Dataset or Model based on its `init_spec()`.
   *
   * @param panel The type of component that will be loaded, either 'Dataset' or
   *    'Model'.
   * @param options The list of components of the type indicated by `panel` that
   *    are loadable, i.e., they provide a non-`null` `init_spec()`.
   * @param selectedOption The name of the component to load.
   * @param spec The `init_spec()` associated with the selected component.
   * @param select A callback function for updating the application state when
   *    the user changes the selected component in the drop-down list. Note that
   *    `this.loadingCallConfig` will always be cleared before the select
   *    function is called.
   * @param load A callback function for telling the LIT server to load a new
   *    instance of the selected component given the configured parameters.
   */
  private renderLoader(panel: string, options: string[], selectedOption: string,
                       spec: Spec, select: EventHandler, load: EventHandler) {
    const disableReset =
        Object.entries(this.loadingCallConfig)
          .map(([name, value]) =>
              name === NEW_NAME_FIELD ? !value : spec[name]?.default === value)
          .reduce((a, b) => a && b, true);
    const disableSubmit = !this.loadingCallConfig[NEW_NAME_FIELD];
    const specEntries = Object.entries(spec);
    const reset = () => {this.resetLoadingCallConfig();};
    const selectionChanged = (e: Event) => {select(e);};

    const configInputs = specEntries.length ?
      Object.entries({
        [NEW_NAME_FIELD]: new StringLitType(),
        ...spec
      }).map(([fieldName, fieldType]) => {
        const value = this.loadingCallConfig[fieldName];
        const updateConfig = (e: Event) => {
          const {value} = e.target as LitInputField;
          this.loadingCallConfig[fieldName] = value;
        };

        const fieldClasses = classMap({
          'missing': this.missingCallConfigFields.includes(fieldName),
          'option-entry': true
        });

        // TODO(b/277252824): Wrap <label> elements in a LitTooltip describing
        // what that feature does.
        return html`<div class=${fieldClasses}>
          <label>${fieldName}:</label>
          <lit-input-field @change=${updateConfig} .type=${fieldType}
            .value=${value} fill-container>
          </lit-input-field>
        </div>`;
      }) :
      `${panel} '${selectedOption}' does not support configurable loading.`;

    return html`<div class="model-dataset-loading">
      <div class="header">Load a new ${panel.toLowerCase()}</div>
      <div class="controls">
        <div class="selector">
          <label for="model-dataset-select">
            Select a base ${panel.toLowerCase()}
          </label>
          <select class="dropdown"
            id="model-dataset-select" name="model-dataset-select"
            @change=${selectionChanged} value=${selectedOption}>
            ${options.map(name =>
              html`<option value=${name} ?selected=${name === selectedOption}>
                ${name}
              </option>`
            )}
          </select>
        </div>
        <div class="options">${configInputs}</div>
        <div class="actions">
          <lit-tooltip tooltipPosition="left"
                      content=${disableSubmit ? LOAD_DISABLED_TXT : ''}>
            <button class="filled-button" slot="tooltip-anchor" @click=${load}
                    ?disabled=${disableSubmit}>
              Load ${panel}
            </button>
          </lit-tooltip>
          <button class="hairline-button" @click=${reset}
                  ?disabled=${disableReset}>
            Reset
          </button>
        </div>
      </div>
    </div>`;
  }

  renderLayoutConfig() {
    const layouts = Object.keys(this.appState.layouts);

    const renderLayoutAreaInfo = (groupLayout: LitTabGroupLayout) => {
      const entries = Object.entries(groupLayout);

      if (entries.length === 0) {
        return html`<div class='info-group'>
          <div class="info-group-subtitle">Unused in this layout</div>
        </div>`;
      }

      return entries.map(([tabName, tabModules]) => {
        // clang-format off
        return html`
          <div class='info-group'>
            <div class='info-group-subtitle'>
              ${tabName}
            </div>
            ${tabModules.map(tabModule => html`
              <div class='indent-line'>
                ${resolveModuleConfig(tabModule).title}
              </div>`)}
          </div>`;
        // clang-format on
      });
    };

    const renderLayoutOption =
        (name: string) => {
          const checked = this.selectedLayout === name;
          // clang-format off
          const selectorHtml = html`
            <mwc-formfield label=${name}>
              <mwc-radio
                name="layouts"
                ?checked=${checked}
                @change=${() => this.selectedLayout = name}>
              </mwc-radio>
            </mwc-formfield>
            `;
          // clang-format on

          const expanderOpen = this.openLayoutKeys.has(name);
          const onExpanderClick = () => {
              this.toggleInSet(this.openLayoutKeys, name);
          };
          const selected = this.selectedLayout === name;
          const disabled = false;

          // The expanded info contains info about the components.
          const {description, left, lower, upper} = this.appState.layouts[name];
          // clang-format off
          const expandedInfoHtml = html`
            <div class='info-group-title'>Left</div>
            ${renderLayoutAreaInfo(left)}
            <div class='info-group-title'>Upper</div>
            ${renderLayoutAreaInfo(upper)}
            <div class='info-group-title'>Lower</div>
            ${renderLayoutAreaInfo(lower)}
          `;
          // clang-format on
          return this.renderLine(
              name, selectorHtml, selected, disabled, expanderOpen,
              onExpanderClick, false, expandedInfoHtml, (description || ''));
        };

    const configListHTML = layouts.map(name => renderLayoutOption(name));
    // clang-format off
    const buttonsHTML = html`${this.nextPrevButton('Dataset', false)}`;
    // clang-format on
    return this.renderConfigPage(
        'Layout', LAYOUT_DESC, configListHTML, buttonsHTML);
  }

  /** Render the "compatible", "selected", or "incompatible" status. */
  private renderStatus(selected = true, disabled = false) {
    const statusIcon = selected ?
        'check_circle' :
        (disabled ? 'warning_amber' : 'check_circle_outline');
    const statusText = selected ?
        SELECTED_TXT :
        (disabled ? INCOMPATIBLE_TXT : COMPATIBLE_TXT);

    const statusClasses = classMap({
      'status': true,
      'selected': selected,
      'error': disabled
    });
    // clang-format off
    return html`
    <div class=${statusClasses}>
      <mwc-icon>
        ${statusIcon}
      </mwc-icon>
      ${statusText}
    </div>`;
    // clang-format on
  }

  private renderLine(
      name: string, selectorHtml: TemplateResult, selected: boolean,
      disabled: boolean, expanderOpen: boolean, onExpanderClick: () => void,
      renderStatus: boolean, expandedInfoHtml: TemplateResult,
      description = '') {
    const expanderIcon =
        expanderOpen ? 'expand_less' : 'expand_more';  // Icons for arrows.

    const classes = classMap({
      'config-line': true,
      'selected': selected,
      'disabled': disabled,
    });

    // In collapsed bar, show the first line only.
    const descriptionPreview = description.split('\n')[0];
    const formattedPreview = getTemplateStringFromMarkdown(descriptionPreview);
    // Make any links clickable.
    const formattedDescription = getTemplateStringFromMarkdown(description);

    const expandedInfoClasses =
        classMap({'expanded-info': true, 'open': expanderOpen});
    const status = renderStatus ? this.renderStatus(selected, disabled) : '';
    return html`
      <div class=${classes}>
        <div class='fixed-third-col'>
          ${selectorHtml}
        </div>
        <div class='flex-col description-preview'>
          ${formattedPreview}
        </div>
        <div class='status-col'>
          ${status}
          <div class=expander>
            <mwc-icon @click=${onExpanderClick}>
              ${expanderIcon}
            </mwc-icon>
          </div>
        </div>
      </div>
      <div class=${expandedInfoClasses}>
        <div class='one-col'>
          <div class='left-offset'>
            ${expandedInfoHtml}
          </div>
        </div>
        <div class='two-col'>
          <div class=info-group-title> Description </div>
          <div class='description-text'>${formattedDescription}</div>
        </div>
      </div>
    `;
  }

  private toggleInSet(set: Set<string>, elt: string) {
    if (set.has(elt)) {
      set.delete(elt);
    } else {
      set.add(elt);
    }
  }

  private nextPrevButton(tab: TabName, next = true) {
    const icon = next ? 'east' : 'west';  // Arrow direction.
    const classes = classMap({
      'hairline-button': true,
      [next ? 'next' : 'prev']: true
    });
    const onClick = () => {
      this.selectedTab = tab;
      this.resetLoadingCallConfig();
    };
    // clang-format off
    return html`
     <button class=${classes} @click=${onClick}>
      <span class='material-icon'>${icon}</span>
      ${tab}
    </button>
    `;
    // clang-format on
  }

  private renderConfigPage(
      title: TabName, description: string, configListHTML: TemplateResult[],
      buttonsHTML: TemplateResult, extraLineHTML?: TemplateResult) {
    return html`
      <div class="config-title">${title}</div>
      <div class="description"> ${description} </div>
      <div class="config-list">
        ${configListHTML}
      </div>
      ${extraLineHTML}
      <div class='prev-next-buttons'>${buttonsHTML} </div>
    `;
  }

  override firstUpdated() {
    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    });
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-global-settings': GlobalSettingsComponent;
  }
}
