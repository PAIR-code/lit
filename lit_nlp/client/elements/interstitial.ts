/**
 * @fileoverview An interstitial graphic to show when a module is empty.
 *
 * @license
 * Copyright 2023 Google LLC
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

// tslint:disable:no-new-decorators
import {html} from 'lit';
import {customElement, property} from 'lit/decorators.js';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './interstitial.css';


/**
 * An interstitial graphic to show when a module is empty.
 */
@customElement('lit-interstitial')
export class LitInterstitial extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  @property({type: String}) headline = '';
  @property({type: String}) imageSrc = 'static/interstitial-select.png';

  override render() {
    // clang-format off
    return html`
      <div class="interstitial">
        ${this.imageSrc ? html`<img src="${this.imageSrc}" />` : null}
        <p>
          ${this.headline ? html`<strong>${this.headline}</strong>` : null}
          <slot></slot>
        </p>
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-interstitial': LitInterstitial;
  }
}