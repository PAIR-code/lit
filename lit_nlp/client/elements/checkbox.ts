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

import {Checkbox} from '@material/mwc-checkbox';
import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import {css, html, LitElement} from 'lit';
import {classMap} from 'lit/directives/class-map';

/**
 * A custom wrapper around the mwc-checkbox which is a) smaller and b) has a
 * built-in label.
 */
@customElement('lit-checkbox')
export class LitCheckbox extends LitElement {
  @property({type: String}) label = '';
  @property({type: Boolean}) checked = false;
  @property({type: Boolean}) indeterminate = false;
  @property({type: Boolean}) disabled = false;
  @property({type: String}) value = '';

  static override get styles() {
    return css`
      :host {
        outline: none;
        --lit-checkbox-label-white-space: normal;
      }

      .wrapper {
        display: flex;
        align-items: center;
        font-size: 13px;
        height: 28px;
        margin-left: -12px;
        margin-right: 4px;
        line-height: 14px;
      }

      .wrapper.disabled {
        opacity: 0.2;
      }

      lit-mwc-checkbox-internal {
        transform: scale(0.7);
        margin-right: -8px;
      }

      .checkbox-label {
        cursor: pointer;
        white-space: var(--lit-checkbox-label-white-space);
      }
    `;
  }

  override render() {
    const handleChange = () => {
      this.checked = !this.checked;
      const changeEvent = new Event('change');
      this.dispatchEvent(changeEvent);
    };

    const wrapperClass = classMap({wrapper: true, disabled: this.disabled});

    return html`
      <div class=${wrapperClass}>
        <lit-mwc-checkbox-internal
          ?checked=${this.checked}
          ?indeterminate=${this.indeterminate}
          ?disabled=${this.disabled}
          value=${this.value}
          @change=${handleChange}
        >
        </lit-mwc-checkbox-internal>
        <span class="checkbox-label" @click=${handleChange}>${this.label}</span>
      </div>
    `;
  }
}


/**
 * Hackily overrides the `shouldRenderRipple` internal web component property
 * to always return false, no matter how it gets updated via internal
 * hover/focus logic.
 */
class MwcCheckboxOverride extends Checkbox {
  // TODO(b/204677206): remove this once we clean up property declarations.
  __allowInstanceProperties = true;  // tslint:disable-line

  constructor() {
    super();
    Object.defineProperty(this, 'shouldRenderRipple', {
      get: () => false,
      set: () => {},
    });
  }
}

customElements.define('lit-mwc-checkbox-internal', MwcCheckboxOverride);

declare global {
  interface HTMLElementTagNameMap {
    'lit-checkbox': LitCheckbox;
  }
}
