/**
 * @fileoverview Element for a pop-up container.
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

import {styles} from './popup_container.css';

/**
 * Popup controls container, anchored by a clickable toggle.
 *
 * Use the 'toggle-anchor' slot for the control; this will be rendered
 * where the <popup-container> element is placed.
 *
 * The remaining content will be rendered in the popup, which is positioned
 * relative to this anchor following the usual rules for position: absolute,
 * and the top, bottom, left, and right attributes specified in CSS as
 * --popup-top, --popup-left, etc.
 *
 * Default position is below and left-aligned; to right-align
 * set --popup-right: 0 in styling for the <popup-container> element.
 *
 * Usage:
 *  <popup-container style=${popupStyle}>
 *    <div slot="toggle-anchor">
 *      <mwc-icon>settings</mwc-icon>
 *    </div>
 *    <div>
 *      // Content goes here
 *    </div>
 *  </popup-container>
 *
 * If you want to have different control elements (such as filled/outlined or
 * up/down arrows) when the panel is open vs. closed, you can use the
 * 'toggle-anchor-open' and 'toggle-anchor-closed' slots instead:
 *
 *  <popup-container style=${popupStyle}>
 *    <div slot="toggle-anchor-closed">
 *      <mwc-icon>expand_more</mwc-icon>
 *    </div>
 *    <div slot="toggle-anchor-open">
 *      <mwc-icon>expand_less</mwc-icon>
 *    </div>
 *    <div>
 *      // Content goes here
 *    </div>
 *  </popup-container>
 *
 * TODO(b/254786211): add DOM and behavioral tests for this element.
 */
@customElement('popup-container')
export class PopupContainer extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  @property({type: Boolean}) expanded = false;

  toggle() {
    this.expanded = !this.expanded;
  }

  private renderPopup() {
    // clang-format off
    return html`
      <div class='popup-outer-holder'>
        <div class='popup-container'>
           <slot></slot>
        </div>
      </div>
    `;
    // clang-format on
  }

  override render() {
    // Show one or the other, depending on state.
    const openClosedSlotName =
        this.expanded ? 'toggle-anchor-open' : 'toggle-anchor-closed';
    // clang-format off
    return html`
        <div class='popup-toggle-anchor' @click=${() => { this.toggle(); }}>
          <slot name="toggle-anchor"></slot>
          <slot name=${openClosedSlotName}></slot>
        </div>
        ${this.expanded ? this.renderPopup() : null}
    `;
    // clang-format off
  }

  protected clickToClose(event: MouseEvent) {
    const path = event.composedPath();
    if (!path.some(elem => elem  === this)) {
      this.expanded = false;
    }
  }

  override updated() {
    const onBodyClick = (event: MouseEvent) => {
      this.clickToClose(event);
    };
    if (this.expanded) {
      document.body.addEventListener(
          'click', onBodyClick, {passive: true, capture: true});
    } else {
      document.body.removeEventListener(
          'click', onBodyClick, {capture: true});
    }
  }

}

declare global {
  interface HTMLElementTagNameMap {
    'popup-container': PopupContainer;
  }
}
