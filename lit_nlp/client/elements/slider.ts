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

import {html, LitElement} from 'lit';
import {customElement, property} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';

import {styles} from './slider.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';


/** A slider with LIT Brand-compliant styles. */
//TODO(b/267161352) rename lit-slider to reflect new text input option.
@customElement('lit-slider')
export class Slider extends LitElement {
  @property({type: Number}) min = 0;
  @property({type: Number}) max = 1;
  @property({type: Number}) step = 0.1;
  @property({type: Number}) value = 0.5;

  private readonly ROUNDING_TIMEOUT_MS = 3000;
  private roundingTimeoutId: number|null = null;

  static override get styles() {
    return [sharedStyles, styles];
  }

  override render() {
    const normalizedValue = (this.value - this.min) / (this.max - this.min);
    const styles = {'background-size': `${normalizedValue * 100}% 100%`};

    const roundValue = () => {
      if (this.roundingTimeoutId) {clearTimeout(this.roundingTimeoutId);}
      this.value = Math.round(this.value / this.step) * this.step;
      this.dispatchEvent(new Event('change'));
    };

    const onInput = (e:Event) => {
      const {value} = e.target as HTMLInputElement;
      if (value === '.') return;
      this.value = Number(value);
    };

    const onInputImmediate = (e: Event) => {
      onInput(e);
      roundValue();
    };

    const onKeypress = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        roundValue();
      } else {
        if (this.roundingTimeoutId) {clearTimeout(this.roundingTimeoutId);}
        this.roundingTimeoutId = setTimeout(
          roundValue, this.ROUNDING_TIMEOUT_MS);
      }
    };

    const throwError = this.value < this.min || this.value > this.max;

    const toolTipContent = throwError ? `Please choose a value between
      ${this.min.toString()} and ${this.max.toString()}.`: ``;

    const renderNumericInput = html`
      <lit-tooltip content=${toolTipContent}>
        <input slot="tooltip-anchor" type="number"
          class="slider-value ${throwError? "error": ""}"
          step="${this.step}" min=${this.min} max=${this.max}
          .value="${this.value.toString()}"
          @input=${onInput}
          @change=${onInputImmediate}
          @keypress=${onKeypress}
          @blur=${roundValue}>
      </lit-tooltip>`;

    // clang-format off
    return html`
      <div class='slider-label start'>${this.min}</div>
      <input type="range" class="slider" style=${styleMap(styles)}
        min="${this.min}" max="${this.max}" step="${this.step}"
        .value="${this.value.toString()}"
        @change=${onInputImmediate}>
      <div class='slider-label'>${this.max}</div>
      ${renderNumericInput}`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-slider': Slider;
  }
}
