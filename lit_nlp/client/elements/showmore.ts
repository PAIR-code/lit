/**
 * @fileoverview A reusable element for collapsing and expanding text for LIT
 *
 * @license
 * Copyright 2023 Google LLC
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

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './showmore.css';

/**
 * A show more element that denotes hidden text that can be expanded on click.
 *
 *
 * Usage:
 *   <lit-showmore
 *      @showmore=${showMore}>
 *   </lit-showmore>
 */
@customElement('lit-showmore')
export class LitShowMore extends ReactiveElement {
    static override get styles() {
      return [sharedStyles, styles];
    }

    @property({type: Boolean}) visible = true;
    @property({type: Number}) hiddenTextLength = 0;

    /**
     * Renders the Show More element with hidden or visible test.
     */
    override render() {
      const onChange = (e:Event) => {
        const event = new CustomEvent('showmore');
        this.dispatchEvent(event);
        e.stopPropagation();
        e.preventDefault();
        this.visible = false;
      };

      /* Non-standard HTML formatting must be preserved to prevent newlines from
      turning into spaces that expand the background of the Show More element.
      */
      // clang-format off
      const renderIcon = html`<span slot="tooltip-anchor"
        class="material-icon-outlined icon-button"
        @click=${onChange}>more_horiz</span>`;

      const renderShowMore =
        html`<lit-tooltip
          content="Show ${this.hiddenTextLength} more characters">
          ${renderIcon}
        </lit-tooltip>`;

      return html`<span class='lit-showmore'>${renderShowMore}</span>`;
      // clang-format on
    }
  }

  declare global {
    interface HTMLElementTagNameMap {
      'lit-showmore': LitShowMore;
    }
  }
