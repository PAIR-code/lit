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
 * LIT App toolbar.
 */

// tslint:disable:no-new-decorators
import '@material/mwc-icon';
import './documentation';
import './global_settings';
import './main_toolbar';

import {MobxLitElement} from '@adobe/lit-mobx';
import {query} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {classMap} from 'lit/directives/class-map';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {datasetDisplayName} from '../lib/types';
import {AppState, ModulesService, SettingsService, StatusService} from '../services/services';

import {app} from './app';
import {styles} from './app_toolbar.css';
import {DocumentationComponent} from './documentation';
import {GlobalSettingsComponent, TabName} from './global_settings';

/**
 * The header/toolbar of the LIT app.
 */
@customElement('lit-app-toolbar')
export class ToolbarComponent extends MobxLitElement {
  @query('lit-documentation') docElement!: DocumentationComponent;
  @query('lit-global-settings') globalSettingsElement!: GlobalSettingsComponent;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly settingsService = app.getService(SettingsService);
  private readonly statusService = app.getService(StatusService);
  private readonly modulesService = app.getService(ModulesService);

  toggleGlobalSettings() {
    if (this.globalSettingsElement === undefined) return;
    if (this.globalSettingsElement.isOpen) {
      this.globalSettingsElement.close();
    } else {
      this.globalSettingsElement.open();
    }
  }

  toggleDocumentation() {
    if (this.docElement === undefined) return;
    if (this.docElement.isOpen) {
      this.docElement.close();
    } else {
      this.docElement.open();
    }
  }

  jumpToSettingsTab(targetTab: TabName) {
    if (this.globalSettingsElement === undefined) return;
    if (this.globalSettingsElement.isOpen &&
        this.globalSettingsElement.selectedTab === targetTab) {
      this.globalSettingsElement.close();
    } else {
      this.globalSettingsElement.selectedTab = targetTab;
      this.globalSettingsElement.open();
    }
  }

  renderStatusAndTitle() {
    const defaultTitle = 'LIT';
    const subtitle = 'Powered by 🔥 LIT';
    let showSubtitle = false;
    let title = defaultTitle;
    if (this.appState.initialized && this.appState.metadata.pageTitle) {
      title = this.appState.metadata.pageTitle;
      if (!title.includes(defaultTitle)) {
        showSubtitle = true;
      }
    }
    const renderFavicon = () => {
      if (this.statusService.hasError) {
        return html`<img src="static/potato.svg" class="status-emoji">`;
      }
      if (showSubtitle) {
        return null;
      }
      return html`<img src="static/favicon.png" class="status-emoji">`;
    };
    // clang-format off
    return html`
      <div id="title-group">
        <a href="https://github.com/PAIR-code/lit/issues/new" target="_blank">
          ${renderFavicon()}
        </a>
        ${title}
        ${showSubtitle ? html`<div class="subtitle">${subtitle}</div>` : null}
      </div>
    `;
    // clang-format on
  }

  renderModelInfo() {
    const compatibleModels =
        Object.keys(this.appState.metadata.models)
            .filter(
                model => this.settingsService.isDatasetValidForModels(
                    this.appState.currentDataset, [model]));

    if (2 <= compatibleModels.length && compatibleModels.length <= 4) {
      // If we have more than one compatible model (but not too many),
      // show the in-line selector.
      const modelChips = compatibleModels.map(name => {
        const isSelected = this.appState.currentModels.includes(name);
        const classes = {
          'headline-button': true,
          'unselected': !isSelected,  // not the same as default; see CSS
          'selected': isSelected,
        };
        const icon = isSelected ? 'check_box' : 'check_box_outline_blank';
        const updateModelSelection = () => {
          const modelSet = new Set<string>(this.appState.currentModels);
          if (modelSet.has(name)) {
            modelSet.delete(name);
          } else {
            modelSet.add(name);
          }
          this.settingsService.updateSettings({'models': [...modelSet]});
          this.requestUpdate();
        };
        // clang-format off
        return html`
          <button class=${classMap(classes)} title="${name}"
            @click=${updateModelSelection}>
            <span class='material-icon'>${icon}</span>
            &nbsp;
            <span class='headline-button-text' title=${name}>${name}</span>
          </button>
        `;
        // clang-format on
      });
      // clang-format off
      return html`
        ${modelChips}
        <button class='headline-button' title="Select model(s)"
          @click=${() => { this.jumpToSettingsTab("Models"); }}>
          <span class='material-icon-outlined'>smart_toy</span>
          &nbsp;<span class='material-icon'>arrow_drop_down</span>
        </button>
      `;
      // clang-format on
    } else {
      // Otherwise, give a regular button that opens the models menu.
      const buttonText = this.appState.currentModels.join(', ');
      // clang-format off
      return html`
        <button class='headline-button' title="Select model(s)"
          @click=${() => { this.jumpToSettingsTab("Models"); }}>
          <span class='material-icon-outlined'>smart_toy</span>
          &nbsp;
          <span class='headline-button-text'>${buttonText}</span>
          &nbsp;
          <span class='material-icon'>arrow_drop_down</span>
        </button>
      `;
      // clang-format on
    }
  }

  renderDatasetInfo() {
    const buttonText = datasetDisplayName(this.appState.currentDataset);
    // clang-format off
    return html`
      <div class='vertical-separator'></div>
      <button class='headline-button' title="Select dataset"
        @click=${() => { this.jumpToSettingsTab("Dataset"); }}>
        <span class='material-icon'>storage</span>
        &nbsp;
        <span class='headline-button-text'>${buttonText}</span>
        &nbsp;
        <span class='material-icon'>arrow_drop_down</span>
      </button>
    `;
    // clang-format on
  }

  renderLayoutInfo() {
    const layouts = Object.keys(this.appState.layouts);
    const currentLayout = this.appState.layoutName;

    if (2 <= layouts.length && layouts.length <= 4) {
      // If we have more than one layout (but not too many),
      // show the in-line selector.
      const layoutChips = layouts.map(name => {
        const isSelected = (name === currentLayout);
        const classes = {
          'headline-button': true,
          'unselected': !isSelected,  // not the same as default; see CSS
          'selected': isSelected,
        };
        const iconClass =
            isSelected ? 'material-icon' : 'material-icon-outlined';
        const updateLayoutSelection = () => {
          this.settingsService.updateSettings({'layoutName': name});
          this.requestUpdate();
        };
        // clang-format off
        return html`
          <button class=${classMap(classes)} title='Select ${name} layout'
            @click=${updateLayoutSelection}>
            <span class=${iconClass}>view_compact</span>
            &nbsp;
            <span class='headline-button-text'>${name}</span>
          </button>
        `;
        // clang-format on
      });
      // clang-format off
      return html`${layoutChips}`;
      // clang-format on
    } else {
      // Otherwise, give a regular button that opens the layouts menu.
      // clang-format off
      return html`
        <button class='headline-button' title="Select UI layout."
          @click=${() => { this.jumpToSettingsTab("Layout"); }}>
          <span class='material-icon'>view_compact</span>
          &nbsp;
          <span class='headline-button-text' title=${currentLayout}>${currentLayout}</span>
          &nbsp;
          <span class='material-icon'>arrow_drop_down</span>
        </button>
      `;
      // clang-format on
    }
  }

  renderConfigControls() {
    // clang-format off
    return html`
      ${this.appState.initialized ? this.renderModelInfo() : null}
      ${this.appState.initialized ? this.renderDatasetInfo() : null}
      <div class='vertical-separator'></div>
      ${this.renderLayoutInfo()}
      <div class='vertical-separator'></div>
      <div title="Configure models, dataset, and UI." id="config">
        <mwc-icon class="icon-button large-icon white-icon icon-margin"
          @click=${this.toggleGlobalSettings}>
          settings
        </mwc-icon>
      </div>
    `;
    // clang-format on
  }

  onClickCopyLink() {
    navigator.clipboard.writeText(this.appState.getBestURL());
  }

  renderRightCorner() {
    // clang-format off
    const docButton = this.appState.metadata != null ?
        html`
          <mwc-icon class="icon-button large-icon white-icon icon-margin"
            title="Documentation"
            @click=${this.toggleDocumentation}>
            help_outline
          </mwc-icon>` : null;
    return html`
      ${docButton}
      <button class='headline-button unbordered' title="Copy link to this page"
        @click=${this.onClickCopyLink}>
        <span class='material-icon'>link</span>
        &nbsp;Copy Link
      </button>
    `;
    // clang-format on
  }

  override render() {
    const doRenderToolbar =
        (this.appState.initialized &&
         !this.modulesService.getSetting('hideToolbar'));
    // TODO(lit-dev): consider using until() directives here to wait on
    // initialization.
    // clang-format off
    return html`
      ${this.appState.initialized ?
        html`<lit-global-settings></lit-global-settings>` : null}
      ${this.appState.metadata != null ?
        html`<lit-documentation></lit-documentation>` : null}
      <div id="at-top">
        <div id="headline">
          <div class="headline-section">
            ${this.renderStatusAndTitle()}
          </div>
          <div class="headline-section">
            ${this.renderConfigControls()}
          </div>
          <div class="headline-section">
            ${this.renderRightCorner()}
          </div>
        </div>
        ${doRenderToolbar? html`<lit-main-toolbar></lit-main-toolbar>` : null}
      </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-app-toolbar': ToolbarComponent;
  }
}
