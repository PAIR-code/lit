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
import {customElement, html, property, TemplateResult} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {action, computed, observable} from 'mobx';
import {NONE_DS_DICT_KEY, NONE_DS_DISPLAY_NAME} from '../lib/types';

import {app} from '../core/lit_app';
import {AppState, SettingsService} from '../services/services';

import {styles} from './global_settings.css';
import {styles as sharedStyles} from './shared_styles.css';

type TabName = 'Models'|'Dataset'|'Layout';
const MODEL_DESC = 'Select models to explore in LIT.';
const DATASET_DESC =
    'Select a compatible dataset to use with the selected models.';
const LAYOUT_DESC = 'Use a preset layout to optimize your workflow';

const SELECTED_TXT = 'Selected';
const COMPATIBLE_TXT = 'Compatible';
const INCOMPATIBLE_TXT = 'Incompatible';

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
  private readonly appState = app.getService(AppState);
  private readonly settingsService = app.getService(SettingsService);

  @observable private selectedDataset: string = '';
  @observable private selectedLayout: string = '';
  @observable private readonly modelCheckboxValues = new Map<string, boolean>();
  @observable private selectedTab: TabName = 'Models';

  // tslint:disable:no-inferrable-new-expression
  @observable private readonly openModelKeys: Set<string> = new Set();
  @observable private readonly openDatasetKeys: Set<string> = new Set();
  @observable private readonly openLayoutKeys: Set<string> = new Set();

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
        <div id="title-bar">Configure LIT</div>
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
      const click = () => this.selectedTab = tab;
      const classes = classMap({tab: true, selected: this.selectedTab === tab});
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
    // TODO(lit-dev): update link when website is live.
    const help =
        'https://github.com/PAIR-code/lit/blob/main/docs/user_guide.md';
    const github = 'https://github.com/PAIR-code/lit';
    return html`
    <div id="links">
      <a href=${github} target="_blank">
        Github
      </a>
      •
      <a href=${help} target="_blank">
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
      info: true,
      disabled: !this.selectedModels.length,
      error: noModelsSelected
    });
    const modelsStr = this.selectedModels.length ?
        this.selectedModels.join(', ') :
        'No models selected';

    const datasetClasses = classMap({info: true, error: !datasetValid});

    return html`
    <div id="bottombar">
      <div id="state">
        <div> selected model(s):
          <span class=${modelClasses}> ${modelsStr} </span>
        </div>
        <div> selected dataset(s):
          <span class=${datasetClasses}>
            ${this.displayName(this.selectedDataset)}
          </span>
        </div>
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
      this.selectedTab = 'Models';
      this.close();
    };

    const submitDisabled = noModelsSelected || !datasetValid;
    return html`
      <div id="buttons-container">
        <div id="buttons">
          <button @click=${cancel}>Cancel</button>
          <button
            class='accent'
            ?disabled=${submitDisabled}
            @click=${submit}>Submit
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
        }
      };
      const renderSelector = () => html`
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
      // clang-format off
      const expandedInfoHtml = html`
        <div class='info-group-title'>
          Dataset Compatibility
        </div>
        ${allDatasets.map((datasetName: string) => {
          const compatible = this.settingsService.isDatasetValidForModels(
            datasetName, [name]);
          const error = !compatible && (this.selectedDataset === datasetName);
          const classes = classMap({compatible, error, 'info-line': true});
          const icon = compatible ? 'check' : (error ? 'warning_amber' : 'clear');
          return html`
            <div class=${classes}>
              <mwc-icon>${icon}</mwc-icon>
              ${this.displayName(datasetName)} 
            </div>`;
        })}`;
      const description = this.appState.metadata.models[name].description;
      return this.renderLine(
          name, renderSelector, selected, disabled, expanderOpen,
          onExpanderClick, false, expandedInfoHtml, description);
    };

    const configListHTML = availableModels.map(name => renderModelSelect(name));
    const buttonsHTML = html`${this.nextPrevButton('Dataset', true)}`;
    return this.renderConfigPage('Models', MODEL_DESC, configListHTML, buttonsHTML);
  }

  /** Render the datasets page content. */
  renderDatasetConfig() {
    const allDatasets = Object.keys(this.appState.metadata.datasets);
    const renderDatasetSelect = (name: string) => {
      const displayName = this.displayName(name);
      const handleDatasetChange = () => {
        this.selectedDataset = name;
      };

      const selected = this.selectedDataset === name;
      const disabled = !this.settingsService.isDatasetValidForModels(
          name, this.selectedModels);

      const expanderOpen = this.openDatasetKeys.has(name);
      const onExpanderClick = () => {
          this.toggleInSet(this.openDatasetKeys, name);
        };
      const renderSelector = () => html`
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

      // Expanded info contains available datasets.
      const allModels = [...this.modelCheckboxValues.keys()];
      // clang-format off
      const expandedInfoHtml = html`
        <div class='info-group-title'>
          Model Compatibility
        </div>
        ${allModels.map((modelName: string) => {
        const compatible = this.settingsService.isDatasetValidForModels(
          name, [modelName]);
        const error = !compatible && (this.selectedModels.includes(modelName));
        const classes = classMap({compatible, error, 'info-line': true});
        const icon = compatible ? 'check' : (error ? 'warning_amber' : 'clear');
        return html`
        <div class=${classes}>
          <mwc-icon>${icon}</mwc-icon>
          ${modelName} 
        </div>`;
      })}`;
      const description = this.appState.metadata.datasets[name].description;
      // clang-format on
      return this.renderLine(
          name, renderSelector, selected, disabled, expanderOpen,
          onExpanderClick, true, expandedInfoHtml, description);
    };
    const configListHTML = allDatasets.map(name => renderDatasetSelect(name));
    // clang-format off
    const buttonsHTML = html`
      ${this.nextPrevButton('Models', false)}
      ${this.nextPrevButton('Layout', true)}
    `;
    // clang-format on

    return this.renderConfigPage(
        'Dataset', DATASET_DESC, configListHTML, buttonsHTML);
  }

  renderLayoutConfig() {
    const layouts = Object.keys(this.appState.layouts);
    const renderLayoutOption =
        (name: string) => {
          const checked = this.selectedLayout === name;
          // clang-format off
          const renderSelector = () => html`
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
          const groups = this.appState.layouts[name].components;
          // clang-format off
          const expandedInfoHtml = html`
          <div class='info-group-title'>
            Modules
          </div>
          ${
            Object.keys(groups).map((groupName: string) => 
              html`
              <div class='info-group'> 
                <div class='info-group-subtitle'>
                  ${groupName}
                </div>
                ${groups[groupName].map(module => 
                  html`<div class='indent-line'>${module.title}</div>`)}
              </div>`
          )}`;
          // clang-format on
          const description = this.appState.layouts[name].description || '';
          return this.renderLine(
              name, renderSelector, selected, disabled, expanderOpen,
              onExpanderClick, false, expandedInfoHtml, description);
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

    const statusClasses = classMap({status: true, selected, error: disabled});
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
      name: string, renderSelector: (name: string) => TemplateResult, selected: boolean,
      disabled: boolean, expanderOpen: boolean, onExpanderClick: () => void,
      renderStatus: boolean, expandedInfoHtml: TemplateResult,
      description = '') {
    const expanderIcon =
        expanderOpen ? 'expand_less' : 'expand_more';  // Icons for arrows.

    const classes = classMap({
      'config-line': true,
      selected,
      disabled,
    });

    const expandedInfoClasses =
        classMap({'expanded-info': true, open: expanderOpen});
    const status = renderStatus ? this.renderStatus(selected, disabled) : '';
    return html`
      <div class=${classes}>
        <div class='one-col'>
          ${renderSelector(name)}
        </div>
        <div class='one-col description-preview'>
         ${description}
        </div>
        <div class='one-col col-end'>
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
          <div>${description}</div>
        </div>
      </div>
    `;
  }

  private displayName(name: string) {
    return name === NONE_DS_DICT_KEY ? NONE_DS_DISPLAY_NAME : name;
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
    const classes = classMap({'next': next, 'prev': !next});
    const onClick = () => this.selectedTab = tab;
    // clang-format off
    return html`
     <button class=${classes} @click=${onClick}>
      <mwc-icon>${icon}</mwc-icon>
      ${tab}
    </button>
    `;
    // clang-format on
  }

  private renderConfigPage(
      title: TabName, description: string, configListHTML: TemplateResult[],
      buttonsHTML: TemplateResult) {
    return html`
      <div class="config-title">${title}</div>
      <div class="description"> ${description} </div>
      <div class="config-list">
        ${configListHTML}
      </div>
      <div class='prev-next-buttons'>${buttonsHTML} </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-global-settings': GlobalSettingsComponent;
  }
}
