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

import '@material/mwc-icon';
import './global_settings';
import './selection_toolbar';
import '../elements/spinner';

import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';

import {app} from '../core/lit_app';
import {AppState, StatusService} from '../services/services';

import {styles} from './app_statusbar.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * The bottom status bar of the LIT app.
 */
@customElement('lit-app-statusbar')
export class StatusbarComponent extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly statusService = app.getService(StatusService);

  render() {
    const progressClass = classMap({
      'progress-line': this.statusService.hasMessage,
      'no-progress-line': !this.statusService.hasMessage
    });

    // clang-format off
    return html`
      <div id="at-bottom">
        <div class="toolbar">
          <div class="text-line">
            <div id="status" class="info">
              ${this.statusService.hasMessage ? this.renderMessages() : null}
            </div>
            <div class="signature">
              <div>Made with <img src="static/favicon.png" class="emoji"> by the LIT team</div>
              <div title="Send feedback" id="feedback">
                <a href="https://github.com/PAIR-code/lit" target="_blank">
                  <mwc-icon class="icon-button">
                    feedback
                  </mwc-icon>
                </a>
              </div>
            </div>
          </div>
          <div class=${progressClass}></div>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderMessages() {
    return html`
      ${this.statusService.hasError ? this.renderError() : this.renderLoading()}
    `;
  }

  renderError() {
    const onClearErrorsClick = () => {
      this.statusService.clearErrors();
    };
    return html`
     <div class="error">${this.statusService.errorMessage}</div>
      <mwc-icon class="icon-button" id="clear-errors" @click=${
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
