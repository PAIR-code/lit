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

import '@material/mwc-switch';

import {css, html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';

/**
 * A custom wrapper around the mwc-switch which includes labels.
 */
@customElement('lit-switch')
export class LitSwitch extends LitElement {
  @property({type: String}) labelLeft = '';
  @property({type: String}) labelRight = '';
  @property({type: Boolean}) selected = false;
  @property({type: Boolean}) disabled = false;

  static override get styles() {
    return [css`
      /* Disable Material ripple */
      .switch-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        min-height: 32px; /* make the actual switch visible */
        overflow-y: clip; /* keep mwc-switch overdraw from messing up layout */
      }

      .switch-container > mwc-switch {
        margin-left: 4px;
        margin-right: 4px;
      }

      .switch-container.disabled {
        color: rgba(60, 64, 67, 0.38);
      }

      mwc-switch {
        /* Revert to old teal color, the new purple is ugly. */
        --mdc-switch-selected-hover-handle-color: teal;
        --mdc-switch-selected-hover-track-color: lightseagreen;
        --mdc-switch-selected-focus-handle-color: teal;
        --mdc-switch-selected-focus-track-color: lightseagreen;
        --mdc-switch-selected-pressed-handle-color: teal;
        --mdc-switch-selected-pressed-track-color: lightseagreen;
        --mdc-switch-selected-handle-color: teal;
        --mdc-switch-selected-track-color: lightseagreen;
        /* Hide the ripple effect. */
        --mdc-switch-selected-focus-state-layer-opacity: 0;
        --mdc-switch-selected-hover-state-layer-opacity: 0;
        --mdc-switch-selected-pressed-state-layer-opacity: 0;
        --mdc-switch-unselected-focus-state-layer-opacity: 0;
        --mdc-switch-unselected-hover-state-layer-opacity: 0;
        --mdc-switch-unselected-pressed-state-layer-opacity: 0;
      }
    `];
  }

  override render() {
    const toggleState = () => {
      this.selected = !this.selected;
      const changeEvent = new Event('change');
      this.dispatchEvent(changeEvent);
    };

    const containerClasses = classMap({
      'switch-container': true,
      'disabled': this.disabled,
      'selected': this.selected
    });

    // prettier-ignore
    return html`
      <div class=${containerClasses} @click=${toggleState}>
        <div class='switch-label label-left'>
          ${this.labelLeft}<slot name="labelLeft"></slot>
        </div>
        <mwc-switch ?selected=${this.selected} ?disabled=${this.disabled}>
        </mwc-switch>
        <div class='switch-label label-right'>
          <slot name="labelRight"></slot>${this.labelRight}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-switch': LitSwitch;
  }
}