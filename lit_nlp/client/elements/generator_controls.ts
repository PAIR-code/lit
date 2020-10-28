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

// tslint:disable:no-new-decorators
import '@material/mwc-icon';
import {customElement, html, LitElement, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {observable} from 'mobx';

import {LitType, Spec} from '../lib/types';
import {styles} from './generator_controls.css';
import {styles as sharedStyles} from '../modules/shared_styles.css';

/**
 * Controls panel for a generator.
 */
@customElement('lit-generator-controls')
export class GeneratorControls extends LitElement {
  @observable @property({type: Object}) spec = {};
  @observable @property({type: String}) name = '';
  settings: {[name: string]: string} = {};
  @property({type: Boolean, reflect: true}) opened = false;

  static get styles() {
    return [sharedStyles, styles];
  }

  render() {
    const generate = () => {
      // Event to be dispatched when a generator is applied.
      const event = new CustomEvent('generator-click', {
        detail: {
          name: this.name,
          settings: this.settings
        }
      });
      this.dispatchEvent(event);
    };

    const collapseIconName = this.opened ? 'expand_less' : 'expand_more';
    const onCollapseClick = () => {
      this.opened = !this.opened;
    };
    const contentClasses = {
      'content': true,
      'minimized': !this.opened
    };
    return html`
      <div class="collapsible" @click=${onCollapseClick}>
        <div class="title">${this.name}</div>
        <mwc-icon class="icon-button min-button">
          ${collapseIconName}
        </mwc-icon>
      </div>
      <div class=${classMap(contentClasses)}>
        ${this.renderControls()}
        <div class="buttons-holder">
          <button class="button" @click=${generate}>Apply</button>
        </div>
      </div>
    `;
  }

  renderControls() {
    const spec = this.spec as Spec;
    return Object.keys(spec).map(name => {

      // Ensure a default value for any of the options provided for setting.
      if (this.settings[name] == null) {
        if (spec[name].vocab) {
          this.settings[name] = spec[name].vocab![0];
        }
        else {
          this.settings[name] = spec[name].default as string;
        }
      }

      return html`
          <div class="control-holder">
            <div class="control-name">${name}</div>
            ${this.renderControl(name, spec[name])}
          </div>`;
    });
  }

  renderControl(name: string, controlType: LitType) {
    if (controlType.vocab) {
      // When provided a vocab, render a dropdown, with the first item selected.
      const updateDropdown = (e: Event) => {
        const select = (e.target as HTMLSelectElement);
        this.settings[name] = controlType.vocab![select?.selectedIndex || 0];
      };
      const options = controlType.vocab.map((option, optionIndex) => {
        return html`
          <option value=${optionIndex}>${option}</option>
        `;
      });
      const defaultValue = controlType.vocab[0];
      return html`<select class="dropdown control" @change=${updateDropdown}
          .value=${defaultValue}>
        ${options}
      </select>`;
    }
    else {
      // Render a text input box.
      const value = this.settings[name] || '';
      const updateText = (e: Event) => {
        const input = e.target! as HTMLInputElement;
        this.settings[name] = input.value;
      };
      return html`<input class="control" type="text" @input=${updateText}
          .value="${value}" />`;
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-generator-controls': GeneratorControls;
  }
}
