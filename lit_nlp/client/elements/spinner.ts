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

import {property} from 'lit/decorators.js';
import {customElement} from 'lit/decorators.js';
import {css, html, LitElement} from 'lit';
import {styleMap} from 'lit/directives/style-map.js';

const TRANSPARENT_COLOR = 'rgba(0,0,0,0)';
const DEFAULT_COLOR = 'var(--app-primary-color)';
const makeBorder = (width: number, color: string) =>
    `solid ${width}px ${color}`;

/**
 * A simple implementation of a shared loading spinner
 */
@customElement('lit-spinner')
export class SpinnerComponent extends LitElement {
  @property({type: Number}) size = 16;
  @property({type: String}) color = '';

  static override get styles() {
    return [
      css`
        #spinner {
          margin-left: 10px;
          border-radius: 50%;
          width: 16px;
          height: 16px;
          animation: spin 2s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `,
    ];
  }

  override render() {
    const size = `${this.size}px`;
    const borderWidth = this.size / 8;
    const color = this.color === '' ? DEFAULT_COLOR : this.color;
    const border = makeBorder(borderWidth, color);
    const borderTop = makeBorder(borderWidth, TRANSPARENT_COLOR);
    const spinnerStyle =
        styleMap({height: size, width: size, border, borderTop});

    return html`
      <div id="spinner" style=${spinnerStyle}></div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-spinner': SpinnerComponent;
  }
}
