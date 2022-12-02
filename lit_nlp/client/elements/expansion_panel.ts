/**
 * @fileoverview A reusable expansion panel element for LIT
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
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';

import {styles} from './expansion_panel.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/** Custom expansion event interface for ExpansionPanel */
export interface ExpansionToggle {
  isExpanded: boolean;
}

/**
 * An element that displays a header with a label and a toggle to expand or
 * collapse the subordinate content.
 */
@customElement('expansion-panel')
export class ExpansionPanel extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable @property({type: String}) label = '';
  @observable @property({type: String}) description = '';
  @observable @property({type: Boolean}) expanded = false;
  @observable @property({type: Boolean}) padLeft = false;
  @observable @property({type: Boolean}) padRight = false;

  override render() {
    const contentPadding = (this.padLeft ? 8 : 0) + (this.padRight ? 16 : 0);
    const styles = styleMap({width: `calc(100% - ${contentPadding}px)`});
    const classes = classMap({
      'expansion-content': true,
      'pad-left': this.padLeft,
      'pad-right': this.padRight
    });

    const toggle = () => {
      this.expanded = !this.expanded;
      const event = new CustomEvent<ExpansionToggle>('expansion-toggle', {
        detail: {isExpanded: this.expanded}
      });
      this.dispatchEvent(event);
    };

    const description = this.description ?? this.label;

    // clang-format off
    return html`
        <div class="expansion-header" @click=${toggle}>
          <div class="expansion-label" title=${description}>${this.label}</div>
          <div class='bar-spacer'></div>
          <div class='bar-content'
            @click=${(e: Event) => { e.stopPropagation(); }}>
            <slot name="bar-content"></slot>
          </div>
          <mwc-icon class="icon-button min-button">
            ${this.expanded ? 'expand_less' : 'expand_more'}
          </mwc-icon>
        </div>
        ${this.expanded ?
            html`<div class=${classes} style=${styles}><slot></slot></div>` :
            null}`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'expansion-panel': ExpansionPanel;
  }
}
