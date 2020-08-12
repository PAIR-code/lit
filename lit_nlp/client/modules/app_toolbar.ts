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

import '@material/mwc-icon';
import './global_settings';
import './selection_toolbar';
import '../elements/spinner';

import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html, property} from 'lit-element';

import {app} from '../core/lit_app';
import {AppState, ModulesService, StatusService} from '../services/services';

import {styles} from './app_toolbar.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * The header/toolbar of the LIT app.
 */
@customElement('lit-app-toolbar')
export class ToolbarComponent extends MobxLitElement {
  @property({attribute: false}) isGlobalSettingsOpen = false;

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly statusService = app.getService(StatusService);
  private readonly modulesService = app.getService(ModulesService);

  private readonly closeGlobalSettings = () => {
    this.isGlobalSettingsOpen = false;
  };

  render() {
    const onSettingsClick = () => {
      this.isGlobalSettingsOpen = !this.isGlobalSettingsOpen;
    };
    const renderToolbar =
        (this.appState.initialized &&
         !this.modulesService.getSetting('hideToolbar'));
    // clang-format off
    return html`
      ${this.appState.initialized ? this.renderGlobalSettings() : null}
      <div id="at-top">
        <div id="toolbar">
          <div id="headline">
            <div class="headline-section">
              <div>
                ${this.statusService.hasError ?
                  html`<img src="static/potato.svg" class="status-emoji">` :
                  html`<img src="static/favicon.png" class="status-emoji">`}
                Language Interpretability Tool
              </div>
            </div>
            <div class="headline-section">
              <div title="Edit models and dataset" id="config">
                <mwc-icon class="icon-button" @click=${onSettingsClick}>
                  settings
                </mwc-icon>
              </div>
            </div>
          </div>
        </div>
        ${renderToolbar? this.renderSelectionToolbar() : null}
      </div>
    `;
    // clang-format on
  }

  renderGlobalSettings() {
    return html`
      <lit-global-settings
        ?isOpen=${this.isGlobalSettingsOpen}
        .close=${this.closeGlobalSettings}
      ></lit-global-settings>
    `;
  }

  renderSelectionToolbar() {
    return html`
      <lit-selection-toolbar></lit-selection-toolbar>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-app-toolbar': ToolbarComponent;
  }
}
