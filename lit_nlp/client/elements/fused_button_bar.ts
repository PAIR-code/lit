/**
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

import './tooltip';

import {html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';

import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './fused_button_bar.css';

/**
 * A text token associated with a weight value
 */
export interface ButtonBarOption {
  text: string;
  tooltipText?: string;
  selected?: boolean;
  onClick?: () => void;
}

/**
 * An element that displays fused buttons in a line, with no gaps and
 * rounded corners only on the first and last buttons.
 */
@customElement('lit-fused-button-bar')
export class FusedButtonBar extends LitElement {
  @property({type: Array}) options: ButtonBarOption[] = [];
  @property({type: String}) label = '';
  @property({type: Boolean}) disabled = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private renderOption(option: ButtonBarOption) {
    const buttonClass = classMap({
      'hairline-button': true,
      'active': Boolean(option.selected),
    });

    // clang-format off
    return html`
      <div class='button-bar-item'>
        <lit-tooltip content=${option.tooltipText ?? ""}>
          <button class=${buttonClass} slot="tooltip-anchor"
           ?disabled=${option.onClick === undefined}
           @click=${option.onClick}>
           ${option.text}
          </button>
        </lit-tooltip>
      </div>`;
    // clang-format on
  }

  override render() {
    // clang-format off
    return html`
      ${this.label ? html`<div class='label'>${this.label}</div>` : null}
      <div class="button-bar-container" ?disabled=${this.disabled}>
        ${this.options.map(o => this.renderOption(o))}
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-fused-button-bar': FusedButtonBar;
  }
}
