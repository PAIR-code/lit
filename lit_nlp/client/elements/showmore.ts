/**
 * @fileoverview A reusable element for collapsing and expanding text for LIT
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

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './showmore.css';

/**
 * A show more element that denotes hidden text that can be expanded on click.
 *
 *
 * Usage:
 *   <lit-showmore
 *      show-more=${showMore}>
 *   </lit-showmore>
 */
@customElement('lit-showmore')
export class LitShowMore extends ReactiveElement {
    static override get styles() {
      return [sharedStyles, styles];
    }
    @property({type: Boolean}) visible = false;

    /**
     * Renders the Show More element with hidden or visible test.
     */
    override render() {
      const onChange = (e:Event) => {
        this.visible = true;
        const event = new CustomEvent('show-more');
        this.dispatchEvent(event);
        e.stopPropagation();
        e.preventDefault();
      };

      return html`
        <div class='lit-showmore'>
          <slot name="showmore-anchor">
            ${this.visible ? '' : html`
              <span class="material-icon-outlined icon-button" @click=${onChange}>
              more_horiz
              </span>`}
          </slot>
        </div>
      `;
    }
  }

  declare global {
    interface HTMLElementTagNameMap {
      'lit-showmore': LitShowMore;
    }
  }
