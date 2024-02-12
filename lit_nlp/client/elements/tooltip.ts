/*
 * @fileoverview A reusable tooltip for LIT
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
import {html} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {getTemplateStringFromMarkdown} from '../lib/utils';

import {styles} from './tooltip.css';

/**
 * A tooltip element that displays on hover and on click.
 *
 * Use the `tooltip-anchor` slot for the control; this will be rendered where
 * the <lit-tooltip> element is placed.
 *
 * Usage:
 *   <lit-tooltip style=${tooltipStyle> .content=${tooltipMarkdown}>
 *     <button slot="tooltip-anchor">
 *     </button>
 *   </lit-tooltip>
 */
@customElement('lit-tooltip')
export class LitTooltip extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  // Markdown that shows on hover.
  @property({type: String}) content = '';
  // Space-separated tooltip position attributes, e.g. 'above' or 'right'.
  @property({type: String}) tooltipPosition = '';
  @property({type: Boolean}) shouldRenderAriaLabel = true;
  @property({type: Boolean}) disabled = false;
  // If we should force-show this, even without mouseover.
  @property({type: Boolean}) forceShow = false;

  renderAriaLabel() {
    return this.shouldRenderAriaLabel ? html`aria-label=${this.content}` : '';
  }

  /**
   * Renders the reference tooltip.
   */
  override render() {
    const tooltipClass = classMap({
      'tooltip-text': true,
      'above': this.tooltipPosition.indexOf('above') !== -1,
      'left': this.tooltipPosition.indexOf('left') !== -1,
      'force-show': this.forceShow,
      'disabled': this.disabled,
    });

    // prettier-ignore
    return html`<div class='lit-tooltip'>
        <slot name="tooltip-anchor">
          ${this.content === '' ? '' : html`
            <span class="help-icon material-icon-outlined icon-button">
              help_outline
            </span>`}
        </slot>
        ${this.content === '' ? '' : html`
          <span class=${tooltipClass} ${this.renderAriaLabel()}>
              ${getTemplateStringFromMarkdown(this.content)}
          </span>`}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-tooltip': LitTooltip;
  }
}
