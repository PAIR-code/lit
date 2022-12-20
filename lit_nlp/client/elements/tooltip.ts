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
import {customElement, property} from 'lit/decorators';

import {styles} from './tooltip.css';
import {ReactiveElement} from '../lib/elements';
import {getTemplateStringFromMarkdown} from '../lib/utils';
import {styles as sharedStyles} from '../lib/shared_styles.css';

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

  /**
   * Renders the reference tooltip with text and optional URL.
   */
  override render() {
    if (this.content === '') {
      return html``;
    }

    return html`
        <div class='lit-tooltip'>
          <span class="help-icon material-icon-outlined icon-button">
            help_outline
          </span>
          <span class='tooltip-text'>
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
