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

/** Removes non-digit chars from a style value and converts it to a number. */
function stylePropToNumber(styles: CSSStyleDeclaration,
                           property: string): number {
  try {
    return Number(styles.getPropertyValue(property).replace(/[^\d\.]/g, ''));
  } catch {
    return 0;
  }
}

/**
 * Color legend visualization component.
 */
@customElement('color-legend')
export class ColorLegend extends ReactiveElement {
  @observable @property({type: Object}) scale: D3Scale =
      d3.scaleOrdinal([DEFAULT]).domain(['all']) as D3Scale;
  @property({type: String}) legendType = LegendType.CATEGORICAL;
  @property({type: String}) selectedColorName = '';
  /** Width of the container. Used to determine if blocks should be labeled. */
  @property({type: Number}) legendWidth = 150;

  // font attributes used to compute whether or not to show the text labels
  private fontFamily: string = '';
  private fontStyle: string = '';
  private fontSize: string = '';

  // label margin values will be updated to be correct one in firstUpdated
  private labelMarginLeft: number = 3;
  private labelMarginRight: number = 2;

  private boxWidth: number = 13;
  private boxMargin: number = 2;

  private selectedColorLabelWidth: number = 46;
  private iconWidth: number = 16;

  static override get styles() {
    return [sharedStyles, styles];
  }

  override firstUpdated() {
    const {host} = this.shadowRoot!;
    if (host) {
      const style = window.getComputedStyle(host);
      this.legendWidth = stylePropToNumber(style, 'width') || this.legendWidth;
    }

    /** Get font styles from the legend-label */
    const legendLabelElement = this.shadowRoot!.querySelector('.legend-label');
    if (legendLabelElement) {
      const style = window.getComputedStyle(legendLabelElement);
      this.fontFamily = style.getPropertyValue('font-family');
      this.fontStyle = style.getPropertyValue('font-style');
      this.fontSize = style.getPropertyValue('font-size');

      this.labelMarginLeft =
          stylePropToNumber(style, 'margin-left') || this.labelMarginLeft;
      this.labelMarginRight =
          stylePropToNumber(style, 'margin-right') || this.labelMarginRight;
    }

    /** Get styles from the legend-box */
    const boxElement = this.shadowRoot!.querySelector('.legend-box');
    if (boxElement) {
      const style = window.getComputedStyle(boxElement);
      this.boxWidth = stylePropToNumber(style, 'width') || this.boxWidth;
      this.boxMargin = stylePropToNumber(style, 'margin') || this.boxMargin;
    }

    /** Get styles from the color-label */
    const colorLabelElement = this.shadowRoot!.querySelector('.color-label');
    if (colorLabelElement) {
      const style = window.getComputedStyle(colorLabelElement);
      const marginLeft = stylePropToNumber(style, 'margin-left') || 3;
      const marginRight = stylePropToNumber(style, 'margin-right') || 3;
      this.selectedColorLabelWidth = marginLeft + marginRight +
          stylePropToNumber(style, 'width') || this.selectedColorLabelWidth;
    }

    /** Get styles from the palette-icon */
     const iconElement = this.shadowRoot!.querySelector('.palette-icon');
     if (iconElement) {
       const style = window.getComputedStyle(iconElement);
       this.iconWidth = stylePropToNumber(style, 'width') || this.iconWidth;
     }
  }

  // TODO(b/237418328): Add a custom tooltip for a faster display time.
  /**
   * Render individual color block and the associated Label
   * Hide the labels if it's a squential legendType or
   * a categorical legendType which width exceeds legendWidth
   */
   private renderLegendBlock(val: string|number, hideLabels: boolean) {
    const background = this.scale(val);
    const style = styleMap({'background': background});

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
   * Render color blocks for sequential values.
   * When hovering over the blocks, a range of mapping values will be displayed
   * @param {string|number} startVal - the min value of a range
   * @param {string|number} endVal - the max value of a range
   * @param {string|number} colorVal - for coloring the block
   * @param {boolean} includeMax - whether to include the max value in a range
   */
  private renderSequentialBlock(startVal: string|number, endVal: number|string,
    colorVal: string|number, includeMax: boolean = false) {
    const title = startVal === endVal ? startVal :
            includeMax ? `[${startVal}, ${endVal}]`
                       : `[${startVal}, ${endVal})`;
    const background = this.scale(colorVal);
    const style = styleMap({'background': background});

    // TODO(b/237418328): Add a custom tooltip for a faster display time.
    // clang-format off
    return html`
      <div class='legend-line'>
        <div class='legend-box' title=${title} style=${style}></div>
      </div>
    `;
    // clang-format on
  }

  /**
   * Render color legend for categorical legend type
   */
  private renderCategoricalLegend() {
    const domain = this.scale.domain();
    const hideLabels = domain.length === 1 ||
                       this.fullLegendWidth > this.legendWidth;
    // clang-format off
    return html`
        <div class="legend-container">
          <mwc-icon class="icon material-icon-outlined">palette</mwc-icon>
          <div class="color-label" title=${this.selectedColorName}
            name="color-name">
            ${this.selectedColorName}
          </div>
          ${this.scale.domain().map(
              (val: string|number) => this.renderLegendBlock(val, hideLabels))}
        </div>
        `;
    // clang-format on
  }

  /**
   * Render color legend for sequential legend type
   */
  private renderSequentialLegend() {
    const [minValue, maxValue] = this.scale.domain() as [number, number];
    const blocks = 7;
    const domain = linearSpace(minValue, maxValue, blocks);
    let curMin = minValue;
    let rangeUnit = (maxValue - minValue) / blocks;
    // round it to an integer if the value is greater than or equal to 5
    rangeUnit = rangeUnit >= 5 ? Math.round(rangeUnit) : rangeUnit;


    // clang-format off
    return html`
        <div class="legend-container">
          <mwc-icon class="icon material-icon-outlined">palette</mwc-icon>
          <div class="color-label" title=${this.selectedColorName}
            name="color-name">
            ${this.selectedColorName}
          </div>
          <div class='legend-label'>${this.toStringValue(minValue)}</div>
          ${domain.map((colorVal: number) => {
            if (colorVal !== minValue) {
              curMin += rangeUnit;
            }

            return this.renderSequentialBlock(
              this.toStringValue(curMin),
              this.toStringValue(
                colorVal === maxValue ? maxValue : curMin + rangeUnit),
              this.toStringValue(colorVal),
              colorVal === maxValue);
          })}
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
