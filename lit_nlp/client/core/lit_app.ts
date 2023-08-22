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
 * Client-side (UI) code for the LIT tool.
 */

import './app_statusbar';
import './app_toolbar';
import './modules';

import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement} from 'lit/decorators.js';
import { html} from 'lit';

import {AppState} from '../services/services';

import {app} from './app';
import {styles} from './lit_app_styles.css';

/**
 * The main LIT app. Contains app-level infrastructure (such as the header,
 * footer, drawers), and renders LIT modules via the `main-page` component.
 */
@customElement('lit-app')
export class AppComponent extends MobxLitElement {
  static override get styles() {
    return [styles];
  }

  private readonly appState = app.getService(AppState);

  override render() {
    return html`
      <lit-app-toolbar></lit-app-toolbar>
      <!-- Main content -->
      ${this.appState.initialized ? html`<lit-modules></lit-modules>` : null}
      <lit-app-statusbar></lit-app-statusbar>
    `;
  }
}
