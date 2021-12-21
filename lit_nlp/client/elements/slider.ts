/**
 * @fileoverview An input[type="range"] slider with LIT Brand-compliant styles
 *
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

import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import {html, LitElement} from 'lit';
import {styleMap} from 'lit/directives/style-map';

import {styles} from './slider.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/** A slider with LIT Brand-compliant styles. */
@customElement('lit-slider')
export class Slider extends LitElement {
  @property({type: Number}) min = 0;
  @property({type: Number}) max = 1;
  @property({type: Number}) step = 0.1;
  @property({type: Number}) val = 0.5;
  @property({attribute: false}) onChange = () => {};
  @property({attribute: false}) onInput = (e:Event) => {
      const input = e.target as HTMLInputElement;
      this.val = +input.value;
  };

  static override get styles() {
    return [sharedStyles, styles];
  }

  override render () {
    const normalizedValue = (this.val - this.min) / (this.max - this.min);
    const styles = {'background-size': `${normalizedValue * 100}% 100%`};
    return html`<input type="range" class="slider" style=${styleMap(styles)}
                  min="${this.min}" max="${this.max}" step="${this.step}"
                  .value="${this.val.toString()}"
                  @input=${this.onInput}
                  @change=${this.onChange}>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-slider': Slider;
  }
}
