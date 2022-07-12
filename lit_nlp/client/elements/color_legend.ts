/**
 * @fileoverview Element for displaying color legend.
 *
 * @license
 * Copyright 2022 Google LLC
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
import * as d3 from 'd3';
import {html} from 'lit';
import {customElement, property} from 'lit/decorators';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {DEFAULT} from '../lib/colors';
import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {D3Scale} from '../lib/types';
import {getTextWidth, linearSpace} from '../lib/utils';

import {styles} from './color_legend.css';

/**
 * Enumeration of the different legend types
 */
export enum LegendType {
  SEQUENTIAL = 'sequential',
  CATEGORICAL = 'categorical'
}

// default width of a character
const DEFAULT_CHAR_WIDTH: number = 5.7;

/**
 * Color legend visualization component.
 */
@customElement('color-legend')
export class ColorLegend extends ReactiveElement {
  @observable
  @property({type: Object})
  scale: D3Scale = d3.scaleOrdinal([DEFAULT]).domain(['all']) as D3Scale;
  @observable @property({type: String}) legendType = LegendType.CATEGORICAL;
  // legendWidth allow the surrounding module to determine the width of a legend
  @observable @property({type: Number}) legendWidth = 150;
  @observable @property({type: String}) selectedColorName = '';
  @property({type: Number}) numBlocks?: number;

  private fontFamily: string = '';
  private fontStyle: string = '';
  private fontSize: string = '';

  // label margin values will be updated to be correct one in firstUpdated
  private labelMarginLeft: number = 3;
  private labelMarginRight: number = 2;

  private boxWidth: number = 13;
  private boxMargin: number = 5;

  private selectedColorLabelWidth: number = 46;
  private iconWidth: number = 16;

  static override get styles() {
    return [sharedStyles, styles];
  }

  override firstUpdated() {
    /**
     * retrieve the font styling information from the legend-label style
     */
    const legendLabelElement = this.shadowRoot!.querySelector('.legend-label');

    if (legendLabelElement) {
      const style = window.getComputedStyle(legendLabelElement);
      this.fontFamily = style.getPropertyValue('font-family');
      this.fontStyle = style.getPropertyValue('font-style');
      this.fontSize = style.getPropertyValue('font-size');

      // get the numerical value only (remove "px")
      this.labelMarginLeft =
          Number(style.getPropertyValue('margin-left').replace(/[^\d]/g, '')) ||
          this.labelMarginLeft;
      this.labelMarginRight =
          Number(
              style.getPropertyValue('margin-right').replace(/[^\d]/g, '')) ||
          this.labelMarginRight;
    }

    /**
     * retrieve the styling information from the legend-box style
     */
    const boxElement = this.shadowRoot!.querySelector('.legend-box');

    if (boxElement) {
      const style = window.getComputedStyle(boxElement);
      this.boxWidth =
          Number(style.getPropertyValue('width').replace(/[^\d]/g, '')) ||
          this.boxWidth;
      this.boxMargin =
          Number(style.getPropertyValue('margin').replace(/[^\d]/g, '')) ||
          this.boxMargin;
    }

    /**
     * retrieve the styling information from the color-label style
     */
    const colorLabelElement = this.shadowRoot!.querySelector('.color-label');

    if (colorLabelElement) {
      const style = window.getComputedStyle(colorLabelElement);
      const marginLeft =
          Number(style.getPropertyValue('margin-left').replace(/[^\d]/g, ''))
          || 3;
      const marginRight =
          Number(style.getPropertyValue('margin-right').replace(/[^\d]/g, ''))
          || 3;
      this.selectedColorLabelWidth = marginLeft + marginRight +
          Number(style.getPropertyValue('width').replace(/[^\\d]/g, '')) ||
          this.selectedColorLabelWidth;
    }

    /**
     * retrieve the styling information from the palette-icon style
     */
     const iconElement = this.shadowRoot!.querySelector('.palette-icon');

     if (iconElement) {
       const style = window.getComputedStyle(iconElement);
       this.iconWidth =
           Number(style.getPropertyValue('width').replace(/[^\d]/g, '')) ||
           this.iconWidth;
     }
  }

  /**
   * Render individual color block and the associated Label
   * Hide the labels if it's a squential legendType or
   * a categorical legendType which width exceeds legendWidth
   */
  private renderLegendBlock(val: string|number) {
    const background = this.scale(val);
    const style = styleMap({'background': background});
    const hideLabels = this.legendType === LegendType.SEQUENTIAL ||
        this.fullLegendWidth > this.legendWidth;

    // clang-format off
    return html`
      <div class='legend-line'>
        <div class='legend-box' title=${val} style=${style}></div>
        <div class='legend-label' ?hidden=${hideLabels}>${val}</div>
      </div>
    `;
    // clang-format on
  }

  /**
   * Render color legend for categorical legend type
   */
  private renderCategoricalLegend() {
    const domain = this.scale.domain();
    const hideLegend =
        domain.length === 1 && domain[0].toString().toLowerCase() === 'all';
    const style = styleMap({'width': `${this.legendWidth}px`});

    // clang-format off
    return html`
        <div class="legend-container" style=${style}>
          <mwc-icon class="palette-icon icon-outlined">palette</mwc-icon>
          <div class="color-label" title=${this.selectedColorName}
            name="color-name">
            ${this.selectedColorName}
          </div>
          ${domain && !hideLegend
            ? domain.map((val: string|number) => this.renderLegendBlock(val))
            : null}
        </div>
        `;
    // clang-format on
  }

  /**
   * Render color legend for sequential legend type
   */
  private renderSequentialLegend() {
    const numDomain = this.scale.domain() as number[];
    const minValue = numDomain ? Math.min(...numDomain) : 0;
    const maxValue = numDomain ? Math.max(...numDomain) : 0;
    const domain = linearSpace(minValue, maxValue, this.numBlocks || 5);

    const style = styleMap({'width': `${this.legendWidth}px`});

    // clang-format off
    return html`
        <div class="legend-container" style=${style}>
          <mwc-icon class="palette-icon icon-outlined">palette</mwc-icon>
          <div class="color-label" title=${this.selectedColorName}
            name="color-name">
            ${this.selectedColorName}
          </div>
          <div class='legend-label'>${this.toStringValue(minValue)}</div>
          ${domain
            ? domain.map((val: number) =>
                this.renderLegendBlock(this.toStringValue(val)))
            : null}
          <div class='legend-label'>${this.toStringValue(maxValue)}</div>
        </div>
        `;
    // clang-format on
  }

  /**
   * Convert a number to a string.
   * Show two digits after the decimal point if the number is not an integer
   */
  private toStringValue(num: number) {
    return num % 1 === 0 ? num.toString() : num.toFixed(2);
  }

  /**
   * Get the approximated width of the legend element
   */
  @computed
  private get fullLegendWidth(): number {
    const domain = this.scale.domain();
    let textWidth = 0;  // label text width

    /**
     * call getTextWidth for text width when all the font information are valid
     * Otherwise, calculate the value using the DEFAULT_CHAR_WIDTH
     */
    if (this.fontFamily && this.fontSize && this.fontStyle) {
      const font = `"'${this.fontFamily}', ${this.fontStyle}"`;
      const fontStyleInfo = `${this.fontSize} ${font}`;
      textWidth = getTextWidth(
          domain.join(''), fontStyleInfo, DEFAULT_CHAR_WIDTH);
    } else {
      textWidth = domain.join('').length * DEFAULT_CHAR_WIDTH;
    }

    // number of blocks * (block width + margain * 2)
    const blocksWidth = domain.length * (this.boxWidth + this.boxMargin * 2);
    // label text width + number of labels * (left margin + right margin)
    const labelsWidth = this.selectedColorLabelWidth + textWidth +
        domain.length * (this.labelMarginLeft + this.labelMarginRight);

    return this.iconWidth + blocksWidth + labelsWidth;
  }

  override render() {
    return this.legendType === LegendType.CATEGORICAL ?
        this.renderCategoricalLegend() :
        this.renderSequentialLegend();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'color-legend': ColorLegend;
  }
}