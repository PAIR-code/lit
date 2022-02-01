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
import '@material/mwc-formfield';
import '@material/mwc-radio';
import '@material/mwc-textfield';
import '../elements/checkbox';

import {MobxLitElement} from '@adobe/lit-mobx';
import {html} from 'lit';
import {customElement, property, query} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {unsafeHTML} from 'lit/directives/unsafe-html.js';

import {marked} from 'marked';
import {computed} from 'mobx';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {AppState} from '../services/services';

import {app} from './app';
import {styles} from './documentation.css';

/**
 * The documentation page.
 */
@customElement('lit-documentation')
export class DocumentationComponent extends MobxLitElement {
  @property({type: Boolean}) isOpen = true;

  static override get styles() {
    return [sharedStyles, styles];
  }
  private readonly appState = app.getService(AppState);

  @query('#holder') docElement!: HTMLDivElement;

  /**
   * Open the documentation dialog.
   */
  open() {
    this.isOpen = true;
  }

  @computed
  get markdownString() {
    return marked(this.appState.metadata.inlineDoc!);
  }

  /**
   * Close the documentation dialog.
   */
  close() {
    this.isOpen = false;
  }

  private markdownHtmlToTemplateString(str: string) {
    return unsafeHTML(str);
  }

  override render() {
    const hiddenClassMap = classMap({hide: !this.isOpen});
    const markdown = this.markdownHtmlToTemplateString(this.markdownString);
    // clang-format off
    return html`
      <div id="doc-holder">
        <div id="overlay" class=${hiddenClassMap}
         @click=${() => { this.close(); }}></div>
        <div id="doc" class=${hiddenClassMap}>
          <div id="title-bar">Welcome to LIT</div>
          <div id="holder">
             ${markdown}
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  override firstUpdated() {
    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Escape') this.close();
    });
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-documentation': DocumentationComponent;
  }
}
