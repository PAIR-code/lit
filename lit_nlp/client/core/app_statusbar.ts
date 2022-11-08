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
 * LIT App toolbar and drawer menu.
 */

// tslint:disable:no-new-decorators
import '@material/mwc-icon';
import './global_settings';
import '../elements/spinner';

import {MobxLitElement} from '@adobe/lit-mobx';
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {observable} from 'mobx';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {AppState, StatusService} from '../services/services';

import {app} from './app';
import {styles} from './app_statusbar.css';

/**
 * The bottom status bar of the LIT app.
 */
@customElement('lit-app-statusbar')
export class StatusbarComponent extends MobxLitElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly statusService = app.getService(StatusService);
  @observable private renderFullMessages = false;

  override render() {
    const progressClass = classMap({
      'progress-line': this.statusService.isLoading,
      'no-progress-line': !this.statusService.isLoading
    });

    // clang-format off
    return html`
      <div id="at-bottom">
        <div class="toolbar">
          <div class="text-line">
            <div class="status-info">
              ${this.statusService.hasMessage ? this.renderMessages() : null}
            </div>
            <div class="signature">
              <div>Made with <img src="static/favicon.png" class="emoji"> by the LIT team</div>
              <div title="Send feedback" id="feedback">
                <a href="mailto:lit-dev@google.com" target="_blank">
                  <mwc-icon class="icon-button cyea-icon">
                    feedback
                  </mwc-icon>
                </a>
              </div>
            </div>
          </div>
          <div class=${progressClass}></div>
        </div>
      </div>
      ${this.renderFullMessages ? this.renderPopup() : null}
    `;
    // clang-format on
  }

  renderMessages() {
    return html`
      ${this.statusService.hasError ? this.renderError() : this.renderLoading()}
    `;
  }

  renderPopupControls() {
    const close = () => {
      this.renderFullMessages = false;
    };

    // clang-format off
    return html`
      <button class='hairline-button' @click=${() => {close();}}>
        Close
      </button>
    `;
    // clang-format on
  }

  renderPopup() {
    // clang-format off
    return html`
      <div class='modal-container'>
        <div class="model-overlay" @click=${() => {close();}}></div>
        <div class='modal'>
          <div class='error-message-header'>Error Details</div>
          <div class='error-message-holder'>
            ${this.statusService.errorFullMessages.map(
                message => html`<div class="error-message">${message}</div>`)}
          </div>
          <div class='close-button-holder'>
            ${this.renderPopupControls()}
          </div>
        </div>
      </div>`;
    // clang-format on
  }

  renderError() {
    const onGetErrorDetailsClick = () => {
      this.renderFullMessages = true;
    };
    const onClearErrorsClick = () => {
      this.statusService.clearErrors();
    };
    return html`
     <div class="error" @click=${onGetErrorDetailsClick}>
        ${this.statusService.errorMessage}
      </div>
      <mwc-icon class="icon-button error" @click=${
        onGetErrorDetailsClick}>
        info
      </mwc-icon>
      <mwc-icon class="icon-button error" @click=${
        onClearErrorsClick}>
        clear
      </mwc-icon>
    `;
  }

  renderLoading() {
    return html`
      <div>${this.statusService.loadingMessage}</div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-app-statusbar': StatusbarComponent;
  }
}
