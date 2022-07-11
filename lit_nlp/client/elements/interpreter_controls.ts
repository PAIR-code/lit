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
import './checkbox';
import '@material/mwc-icon';

import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {LitType, Spec} from '../lib/types';
import {tryCastAsType} from '../lib/lit_types_utils';
import {isLitSubtype} from '../lib/utils';

import {styles} from './interpreter_controls.css';

/** Settings for an interpreter */
export interface InterpreterSettings {
  [name: string]: boolean | number | string | string[];
}

/** Custom click event for interpreter controls */
export interface InterpreterClick {
  name: string;
  settings: InterpreterSettings;
}

/**
 * Controls panel for an interpreter.
 */
@customElement('lit-interpreter-controls')
export class InterpreterControls extends ReactiveElement {
  @observable @property({type: Object}) spec = {};
  @observable @property({type: String}) name = '';
  @observable @property({type: String}) description = '';
  @observable @property({type: String}) applyButtonText = 'Apply';
  @property({type: Boolean, reflect: true}) applyButtonDisabled = false;
  @observable settings: InterpreterSettings = {};
  @property({type: Boolean, reflect: true}) opened = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  override render() {
    const apply = () => {
      // Event to be dispatched when an interpreter is applied.
      const event = new CustomEvent<InterpreterClick>('interpreter-click', {
        detail: {
          name: this.name,
          settings: this.settings
        }
      });
      this.dispatchEvent(event);
    };

    const expandable =
        Object.keys(this.spec).length > 0 || this.description.length > 0;

    // clang-format off
    const content = html`
        <div class="content">
          <div class="description">${this.description}</div>
          ${this.renderControls()}
          <div class="buttons-holder">
            <button class="filled-button" @click=${apply}
              ?disabled=${this.applyButtonDisabled}>
              ${this.applyButtonText}
            </button>
          </div>
        </div>`;

    return expandable ?
        html`<expansion-panel .label=${this.name} ?expanded=${this.opened}>
                ${content}
              </expansion-panel>` :
        html`<div class="header">
                <div class="title">${this.name}</div>
              </div>
              ${this.opened ? content : null}`;
    // clang-format on
  }

  renderControls() {
    const spec = this.spec as Spec;
    return Object.keys(spec).map(name => {
      // Ensure a default value for any of the options provided for setting.
      if (this.settings[name] == null) {
        const litType = tryCastAsType(spec[name], [
          'SparseMultilabel',
          'MultiFieldMatcher',
          'CategoryLabel',
          'FieldMatcher',
        ]);

        if (isLitSubtype(litType, 'SparseMultilabel')) {
          this.settings[name] = litType.default as string[];
        }
        // If select all is True, default value is all of vocab.
        if (isLitSubtype(litType, 'MultiFieldMatcher')) {
          this.settings[name] = litType.select_all ?
              litType.vocab as string[] :
              litType.default as string[];
        }
        // FieldMatcher has its vocab set outside of this element.
        else if (isLitSubtype(litType, ['CategoryLabel', 'FieldMatcher'])) {
          this.settings[name] =
              litType.vocab != null && litType.vocab.length > 0 ?
              litType.vocab[0] :
              '';
        } else {
          this.settings[name] = litType.default as string;
        }
      }
      const required = spec[name].required;
      return html`
          <div class="control-holder">
            <div class="control-name">
              ${(required ? '*' : '') + name}
            </div>
            ${this.renderControl(name, spec[name])}
          </div>`;
    });
  }

  renderControl(name: string, controlType: LitType) {
    const litType = tryCastAsType(controlType, [
      'SparseMultilabel', 'MultiFieldMatcher', 'CategoryLabel', 'FieldMatcher',
      'Scalar'
    ]);
    if (isLitSubtype(litType, ['SparseMultilabel', 'MultiFieldMatcher'])) {
      // Render checkboxes, with the first item selected.
      const renderCheckboxes = () => (litType.vocab as string[]).map(option => {
        // tslint:disable-next-line:no-any
        const change = (e: any) => {
          if (e.target.checked) {
            (this.settings[name] as string[]).push(option);
          } else {
            this.settings[name] = (this.settings[name] as string[])
                                      .filter(item => item !== option);
          }
        };
        const isSelected =
            (this.settings[name] as string[]).indexOf(option) !== -1;
        return html`
          <lit-checkbox ?checked=${isSelected} @change=${change}
            label=${option} class='checkbox-control'>
          </lit-checkbox>
        `;
      });
      return html`<div class='checkbox-holder'>${renderCheckboxes()}</div>`;
    } else if (isLitSubtype(litType, ['CategoryLabel', 'FieldMatcher'])) {
      // Render a dropdown, with the first item selected.
      const updateDropdown = (e: Event) => {
        const select = (e.target as HTMLSelectElement);
        this.settings[name] = litType.vocab[select?.selectedIndex || 0];
      };
      const options = (litType.vocab as string[]).map((option, optionIndex) => {
        return html`
          <option value=${optionIndex}>${option}</option>
        `;
      });
      const defaultValue = litType.vocab != null && litType.vocab.length > 0 ?
          litType.vocab[0] :
          '';
      return html`<select class="dropdown control" @change=${updateDropdown}
          .value=${defaultValue} ?disabled=${litType.vocab.length < 2}>
        ${options}
      </select>`;
    } else if (isLitSubtype(litType, ['Scalar'])) {
      // Render a slider.
      const step = litType.step;
      const minVal = litType.min_val;
      const maxVal = litType.max_val;

      const updateSettings = (e: Event) => {
        const input = (e.target as HTMLInputElement);
        this.settings[name] = input.value;
      };

      // clang-format off
      return html`
        <div class='slider-holder'>
          <div class='slider-label slider-label-start'>${minVal}</div>
          <lit-slider class="slider"
                      min="${minVal}" max="${maxVal}" step="${step}"
                      val="${+this.settings[name]}"
                      .onInput=${updateSettings}></lit-slider>
          <div class='slider-label'>${maxVal}</div>
          <div class='slider-value'>${this.settings[name]}</div>
        </div>
      `;
      // clang-format on
    } else if (isLitSubtype(controlType, ['Boolean'])) {
      // Render a checkbox.
      const toggleVal = () => {
        const val = !!this.settings[name];
        this.settings[name] = !val;
      };
      // clang-format off
      return html`
        <lit-checkbox
         ?checked=${!!this.settings[name]}
         @change=${toggleVal}>
        </lit-checkbox>
      `;
      // clang-format on
    } else if (isLitSubtype(controlType, ['Tokens'])) {
      // Render a text input box and split on commas.
      const value = this.settings[name] as string || '';
      const updateText = (e: Event) => {
        const input = e.target! as HTMLInputElement;
        this.settings[name] = input.value.split(',').map(val => val.trim());
      };
      return html`<input class="control" type="text" @input=${updateText}
          .value="${value}" />`;
    } else {
      // Render a text input box.
      const value = this.settings[name] as string || '';
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
    'lit-interpreter-controls': InterpreterControls;
  }
}
