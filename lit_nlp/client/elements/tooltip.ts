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
import {html, TemplateResult} from 'lit';
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {getTemplateStringFromMarkdown} from '../lib/utils';

import {styles} from './tooltip.css';

type Callback = (e: null) => void;

const DEFAULT_HOVER_HTML =
    html`<span class="help-icon material-icon-outlined icon-button">help_outline</span>`;

/**
 * An element that displays a header with a label and a toggle to expand or
 * collapse the subordinate content.
 */
@customElement('lit-tooltip')
export class LitTooltip extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  // Markdown that shows on hover.
  @property({type: String}) content = '';

  // If no hover icon HTML is provided, we default to an info icon.
  @property({type: Object}) hoverElementHtml?: TemplateResult;
  @property({type: String}) tooltipPosition: string = '';
  @property({type: Boolean}) shouldRenderAriaTitle = true;

  // Callbacks.
  @property({type: Object}) onClick?: Callback;
  @property({type: Object}) onHover?: Callback;

  // If false, don't allow sticky tooltips.
  @property({type: Boolean}) enableSticky = true;
  // Toggle tooltip visibility on click.
  @property({type: Boolean}) isTooltipSticky = false;
  // Keep tooltip visible on hover.
  @property({type: Boolean}) isHovering = false;

  renderAriaTitle() {
    return this.shouldRenderAriaTitle ? html`aria-title=${this.content}` : '';
  }

  /**
   * Renders the reference tooltip.
   */
  override render() {
    // If there's no tooltip content, don't render the element as a tooltip.
    if (this.content === '') {
      return this.hoverElementHtml || html``;
    }

    const displayClass = classMap({
      'hidden': !(this.isHovering ? true : this.isTooltipSticky),
      'tooltip-text': true,
      'above': this.tooltipPosition === 'above',
    });

    const onClick = (e: Event) => {
      if (this.enableSticky) {
        this.isTooltipSticky = !this.isTooltipSticky;
      }
    };

    const onMouseEnter = (e: Event) => {
      this.isHovering = true;
    };

    const onMouseLeave = (e: Event) => {
      this.isHovering = false;
    };

    return html`
      <div class='lit-tooltip' @click=${onClick} @mouseenter=${
        onMouseEnter} @mouseleave=${onMouseLeave}>
        ${this.hoverElementHtml || DEFAULT_HOVER_HTML}
        <span class=${displayClass} ${this.renderAriaTitle()}>
            ${getTemplateStringFromMarkdown(this.content)}
        </span>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-tooltip': LitTooltip;
  }
}
