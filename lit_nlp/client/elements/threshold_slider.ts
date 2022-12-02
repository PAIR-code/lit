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

import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import {css, html, LitElement} from 'lit';
import {classMap} from 'lit/directives/class-map';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {getMarginFromThreshold, getThresholdFromMargin} from '../lib/utils';

/** Custom change event for ThresholdSlider. */
export interface ThresholdChange {
  label: string;
  margin: number;
}

/** A slider for setting classifcation thresholds/margins. */
@customElement('threshold-slider')
export class ThresholdSlider extends LitElement {
  @property({type: Number}) margin = 0;
  @property({type: String}) label = '';
  // Threshold sliders are between 0 and 1 for binary classification thresholds.
  // The other type of sliders are margin sliders between -5 and 5 for use with
  // mutliclass classifiers.
  @property({type: Boolean}) isThreshold = true;
  @property({type: Boolean}) showControls = false;

  static override get styles() {
    return [sharedStyles, css`
        .slider-row {
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: center;
        }

        .text-with-controls {
          padding-top: 0;
        }

        .text-no-controls {
          padding-top: 2px;
        }

        .slider-val {
          color: var(--lit-neutral-600);
          margin-top: -3px; /*Accounts for custom thumb offset in lit-slider*/
          margin-left: 2px;
          width: 30px;
        }
    `];
  }

  renderThresholdSlider(margin: number, label: string) {
    // Convert between margin and classification threshold when displaying
    // margin as a threshold, as is done for binary classifiers.
    // Threshold is between 0 and 1 and represents the minimum score of the
    // positive (non-null) class before a datapoint is classified as positive.
    // A margin of 0 is the same as a threshold of .5 - meaning we take the
    // argmax class. A negative margin is a threshold below .5. Margin ranges
    // from -5 to 5, and can be converted the threshold through the equation
    // margin = ln(threshold / (1 - threshold)).
    const onChange = (e: Event) => {
      const newThresh = +(e.target as HTMLInputElement).value;
      const newMargin = getMarginFromThreshold(newThresh);
      const event = new CustomEvent<ThresholdChange>('threshold-changed', {
        detail: {
          label,
          margin: newMargin
        }
      });
      this.dispatchEvent(event);
    };
    function marginToVal(margin: number) {
      const val = getThresholdFromMargin(+margin);
      return Math.round(100 * val) / 100;
    }
    return this.renderSlider(
        margin, label, 0, 1, 0.01, onChange, marginToVal, 'Threshold');
  }

  renderMarginSlider(margin: number, label: string) {
    const onChange = (e: Event) => {
      const newMargin = (e.target as HTMLInputElement).value;
      const event = new CustomEvent<ThresholdChange>('threshold-changed', {
        detail: {
          label,
          margin: Number(newMargin)
        }
      });
      this.dispatchEvent(event);
    };
    function marginToVal (margin: number) {return margin;}
    return this.renderSlider(
        margin, label, -5, 5, 0.05, onChange, marginToVal, 'margin');
  }

  renderSlider(
      margin: number, label: string, min: number, max: number,
      step: number, onChange: (e: Event) => void,
      marginToVal: (margin: number) => number, title: string) {
    const val = marginToVal(margin);
    const isDefaultValue = margin === 0;

    const reset = () => {
      const event = new CustomEvent<ThresholdChange>('threshold-changed', {
        detail: {
          label,
          margin: 0
        }
      });
      this.dispatchEvent(event);
    };
    const labelClasses = {
      'text-with-controls': this.showControls,
      'text-no-controls': !this.showControls
    };
    const valClasses = {
      'text-with-controls': this.showControls,
      'text-no-controls': !this.showControls,
      'slider-val': true
    };

    const renderLabel = () => {
      if (this.showControls) {
        return html`
          <div class=${classMap(labelClasses)}>${label} ${title}:</div>`;
      } else {
        return null;
      }
    };

    return html`
        <div class="slider-row">
          ${renderLabel()}
          <lit-slider min="${min}" max="${max}" step="${step}" val="${val}"
                      .onChange=${onChange}></lit-slider>
          <div class=${classMap(valClasses)}>${val}</div>
          ${this.showControls ?
              html`<button class='hairline-button reset-button' @click=${reset}
                   ?disabled="${isDefaultValue}">Reset</button>` : null}
        </div>`;
  }

  override render() {
    return html`${this.isThreshold ?
        this.renderThresholdSlider(this.margin, this.label) :
        this.renderMarginSlider(this.margin, this.label)}`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'threshold-slider': ThresholdSlider;
  }
}
