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
import './global_settings';
import './main_toolbar';

import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html, query} from 'lit-element';

import {app} from '../core/lit_app';
import {datasetDisplayName} from '../lib/types';
import {copyToClipboard} from '../lib/utils';
import {AppState, ModulesService, StatusService} from '../services/services';

import {styles} from './app_toolbar.css';
import {GlobalSettingsComponent, TabName} from './global_settings';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * The header/toolbar of the LIT app.
 */
@customElement('lit-app-toolbar')
export class ToolbarComponent extends MobxLitElement {
  @query('lit-global-settings') globalSettingsElement!: GlobalSettingsComponent;

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
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

  jumpToSettingsTab(targetTab: TabName) {
    if (this.globalSettingsElement === undefined) return;
    this.globalSettingsElement.selectedTab = targetTab;
    this.globalSettingsElement.open();
  }

  onCopyLinkClick() {
    const urlBase =
        (this.appState.metadata.canonicalURL || window.location.host);
    copyToClipboard(urlBase + window.location.search);
  }

  renderModelAndDatasetInfo() {
    const modelsPrefix =
        this.appState.currentModels.length > 1 ? 'Models' : 'Model';
    const modelsText = html`
        ${modelsPrefix}:
        <span class='status-text-underline'>
          ${this.appState.currentModels.join(', ')}
        </span>`;
    const datasetText = html`
        Dataset:
        <span class='status-text-underline'>
          ${datasetDisplayName(this.appState.currentDataset)}
        </span>`;
    // clang-format off
    return html`
      <div id='models-data-status'>
        <div class='status-item'
             @click=${() => { this.jumpToSettingsTab("Models"); }}>
          ${modelsText}
        </div>
        <div class='status-item'
             @click=${() => { this.jumpToSettingsTab("Dataset"); }}>
          ${datasetText}
        </div>
      </div>
    `;
    // clang-format on
  }


  render() {
    const doRenderToolbar =
        (this.appState.initialized &&
         !this.modulesService.getSetting('hideToolbar'));
    // TODO(lit-dev): consider using until() directives here to wait on
    // initialization.
    // clang-format off
    return html`
      ${this.appState.initialized ?
        html`<lit-global-settings></lit-global-settings>` : null}
      <div id="at-top">
        <div id="toolbar">
          <div id="headline">
            <div class="headline-section">
              <div>
                <a href="https://github.com/PAIR-code/lit/issues/new" target="_blank">
                  ${this.statusService.hasError ?
                    html`<img src="static/potato.svg" class="status-emoji">` :
                    html`<img src="static/favicon.png" class="status-emoji">`}
                </a>
                ${this.appState.initialized && this.appState.metadata.pageTitle ?
                  this.appState.metadata.pageTitle : "Language Interpretability Tool"}
              </div>
              ${this.appState.initialized ? this.renderModelAndDatasetInfo() : null}
            </div>
            <div class="headline-section">
              <div title="Copy link to this page" id="share">
                <mwc-icon class="icon-button" @click=${this.onCopyLinkClick}>
                  link
                </mwc-icon>
              </div>
              <div title="Change layout" id="layout-button">
                <mwc-icon class="icon-button"
                  @click=${() => { this.jumpToSettingsTab("Layout"); }}>
                  view_compact
                </mwc-icon>
              </div>
              <div title="Edit models and dataset" id="config">
                <mwc-icon class="icon-button"
                  @click=${this.toggleGlobalSettings}>
                  settings
                </mwc-icon>
              </div>
            </div>
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
