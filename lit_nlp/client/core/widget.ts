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
 * Client-side (UI) code for the LIT tool.
 */

// tslint:disable:no-new-decorators
import '../elements/spinner';
import '@material/mwc-icon';

import {MobxLitElement} from '@adobe/lit-mobx';
import {css, customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {observable} from 'mobx';

import {styles} from './widget.css';

/**
 * A wrapper for a LIT module that renders the contents in a box with
 * expand/contract capabilities.
 */
@customElement('lit-widget')
export class LitWidget extends MobxLitElement {
  @property({type: String}) displayTitle = '';
  @property({type: String}) subtitle = '';
  @property({type: Boolean}) isLoading = false;
  @property({type: Boolean}) highlight = false;
  @observable maximized = false;

  static get styles() {
    return styles;
  }

  render() {
    const contentClasses = classMap({
      content: true,
      loading: this.isLoading,
    });
    const holderClasses = classMap({
      holder: true,
      maximized: this.maximized,
      highlight: this.highlight,
    });
    const headerClasses = classMap({
      header: true,
      highlight: this.highlight,
    });

    const contentStyle: {[name: string]: string} = {};
    const holderStyle: {[name: string]: string} = {};

    if (this.maximized) {
      const holderHeight = window.innerHeight - 120;
      const holderWidth = window.innerWidth - 109;
      const contentHeight = holderHeight - 70;

      holderStyle['width'] = `${holderWidth}px`;
      holderStyle['height'] = `${holderHeight}px`;
      contentStyle['minHeight'] = `${contentHeight}px`;
      contentStyle['maxHeight'] = `${contentHeight}px`;
    }

    const iconName = this.maximized ? 'fullscreen_exit' : 'fullscreen';
    const onMaximizeClick = () => {
      this.maximized = !this.maximized;
    };

    // clang-format off
    return html`
      <div class=${holderClasses} style=${styleMap(holderStyle)}>
        <div class=${headerClasses} title=${this.displayTitle}>
          <div>
            <span class="title">${this.displayTitle}</span>
            <span class="subtitle">${this.subtitle}</span>
          </div>
          <mwc-icon class="icon-button" @click=${onMaximizeClick}>
            ${iconName}
          </mwc-icon>
        </div>
        <div class="container">
          ${this.isLoading ? this.renderSpinner() : null}
          <div class=${contentClasses} style=${styleMap(contentStyle)}>
            <slot></slot>
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${36} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-widget': LitWidget;
  }
}
