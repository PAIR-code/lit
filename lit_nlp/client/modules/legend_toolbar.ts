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
 * LIT App legend toolbar
 */

// tslint:disable:no-new-decorators
import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html} from 'lit-element';
import {styleMap} from 'lit-html/directives/style-map';

import {app} from '../core/lit_app';
import {D3Scale} from '../lib/types';
import {range} from '../lib/utils';
import {ColorService} from '../services/services';

import {styles} from './legend_toolbar.css';
import {styles as sharedStyles} from './shared_styles.css';


/**
 * The selection controls toolbar
 */
@customElement('lit-legend-toolbar')
export class LitLegendToolbar extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly colorService = app.getService(ColorService);

  render() {
    const options = this.colorService.colorableOptions;
    const htmlOptions = options.map((option, optionIndex) => {
      const selected =
          this.colorService.selectedColorOption.name === option.name;
      return html`<option value=${optionIndex} ?selected=${selected}>
          ${option.name}</option>`;
    });

    const handleChange = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      const index = select?.selectedIndex || 0;
      this.colorService.selectedColorOption =
          this.colorService.colorableOptions[index];
    };

    const domain = this.colorService.selectedColorOption.scale.domain();
    const numericScale = typeof domain[0] === 'number';

    const renderLegend = () => {
      if (numericScale) {
        return this.renderNumericScale(
            this.colorService.selectedColorOption.scale);
      } else {
        return this.renderCategoricalScale();
      }
    };

    return html`
      <div class="top-holder">
        ${renderLegend()}
        <div class="dropdown-container">
          <label class="dropdown-label">Color datapoints by</label>
          <select class="dropdown color-dropdown" @change=${handleChange}>
            ${htmlOptions}
          </select>
        </div>
      </div>
      `;
  }

  renderCategoricalScale() {
    const renderLegendLine = ((val: string|number) => {
      const background = this.colorService.selectedColorOption.scale(val);
      const style = styleMap({background});
      return html`
        <div class='legend-line'>
          <div class='legend-box' style=${style}></div>
          <div class='dropdown-label'>${val}</div>
        </div>
      `;
    });

    const domain = this.colorService.selectedColorOption.scale.domain();
    // clang-format off
    return html`
        <div class="line-holder">
          ${domain.map((val: string|number) => renderLegendLine(val))}
        </div>
        `;
    // clang-format on
  }

  renderNumericScale(scale: D3Scale) {
    const width = 300;
    const widthArray = range(width);
    const numTicks = 5;
    const ticksArray = range(numTicks);
    const domain = scale.domain() as number[];
    const domainSize = domain[1] - domain[0];
    // clang-format off
    return html`
      <div class='numeric-legend-holder'>
        <div class='numeric-legend'>
          ${widthArray.map((i: number) => {
            const background = scale(domain[0] + i / width * domainSize);
            const style = styleMap({background});
            return html`<div class='legend-pixel' style=${style}></div>`;
          })}
        </div>
        <div class='tick-legend'>
        ${ticksArray.map((i: number) => {
          const left = i / (numTicks - 1) * width;
          const style = styleMap({left: `${left}px`, position: 'absolute'});
          const val = domain[0] + i / (numTicks - 1) * domainSize;
          return html`
              <div class='tick-holder' style=${style}>
                <div class='tick'></div>
                <div class='tick-label'>${val.toFixed(2)}</div>
              </div>`;
        })}
        </div>
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-legend-toolbar': LitLegendToolbar;
  }
}
