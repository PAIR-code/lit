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
import '../elements/numeric_input';

import {html} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {BooleanLitType, CategoryLabel, LitType, LitTypeWithVocab, MultiFieldMatcher, Scalar, SingleFieldMatcher, SparseMultilabel, Tokens} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {type Spec} from '../lib/types';
import {getTemplateStringFromMarkdown} from '../lib/utils';

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
  @observable.struct @property({type: Object}) spec: Spec = {};

  @property({type: String}) name = '';
  @property({type: String}) description = '';
  @property({type: String}) applyButtonText = 'Apply';
  @property({type: Boolean, reflect: true}) applyButtonDisabled = false;
  @property({type: Boolean, reflect: true}) noexpand = false;
  @property({type: Boolean, reflect: true}) opened = false;

  private settings: InterpreterSettings = {};

  static override get styles() {
    return [sharedStyles, styles];
  }

  // TODO(b/295346190): Extract this to utils.ts, add tests, use across the app.
  @computed get defaultSettings(): InterpreterSettings {
    const settings: InterpreterSettings = {};
    for (const [fieldName, fieldSpec] of Object.entries(this.spec)) {
      if (fieldSpec instanceof MultiFieldMatcher) {
        // If select all is True, default value is all of vocab.
        settings[fieldName] = fieldSpec.select_all ?
            (fieldSpec.vocab || []) : fieldSpec.default;
      } else if (
          fieldSpec instanceof CategoryLabel ||
          fieldSpec instanceof SingleFieldMatcher) {
        // SingleFieldMatcher has its vocab set outside of this element.
        const {vocab} = fieldSpec as LitTypeWithVocab;
        if (vocab == null) {
          settings[fieldName] = '';
        } else {
          const [first] = vocab;
          settings[fieldName] = first || '';
        }
      } else if (fieldSpec instanceof SparseMultilabel) {
        settings[fieldName] = fieldSpec.default || [];
      } else {
        settings[fieldName] = fieldSpec.default as string;
      }
    }
    return settings;
  }

  override connectedCallback() {
    super.connectedCallback();
    this.settings = structuredClone(this.defaultSettings);
    this.react(
        () => this.spec,
        () => {this.settings = structuredClone(this.defaultSettings);});
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

    // TODO(b/254775471): revisit this logic, remove need for noexpand
    // in favor of an explicit 'expandable' attribute.
    const expandable = !this.noexpand &&
        (Object.keys(this.spec).length > 0 || this.description.length > 0);

    const descriptionHTML = getTemplateStringFromMarkdown(this.description);

    // clang-format off
    const content = html`
        <div class="content">
          <div class="description">${descriptionHTML}</div>
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
    return Object.entries(this.spec).map(([fieldName, fieldSpec]) =>
      html`<div class="control-holder">
        <div class="control-name">
          ${(fieldSpec.required ? '*' : '') + fieldName}
        </div>
        ${this.renderControl(fieldName, fieldSpec)}
      </div>`);
  }

  renderControl(name: string, controlType: LitType) {
    if (controlType instanceof SparseMultilabel ||
        controlType instanceof MultiFieldMatcher) {
      const {vocab} = controlType as LitTypeWithVocab;
      if (vocab == null) {
        console.error(
            `Cannot render checkboxes for field ${name} without a vocab.`);
        return null;
      }
      // Render checkboxes, with the first item selected.
      const renderCheckboxes = () => vocab.map(option => {
        const change = (e: Event) => {
          if ((e.target as HTMLInputElement).checked) {
            (this.settings[name] as string[]).push(option);
          } else {
            this.settings[name] =
                (this.settings[name] as string[]).filter(e => e !== option);
          }
        };
        const isSelected =
            (this.settings[name] as string[]).indexOf(option) !== -1;
        return html`<lit-checkbox ?checked=${isSelected} @change=${change}
          label=${option} class='checkbox-control'>
        </lit-checkbox>`;
      });
      return html`<div class='checkbox-holder'>${renderCheckboxes()}</div>`;
    } else if (
        controlType instanceof CategoryLabel ||
        controlType instanceof SingleFieldMatcher) {
      const {vocab} = controlType as LitTypeWithVocab;
      if (vocab == null) {
        console.error(
            `Cannot render dropdown for field ${name} without a vocab.`);
        return null;
      }
      // Render a dropdown, with the first item selected.
      const updateDropdown = (e: Event) => {
        const index = (e.target as HTMLSelectElement).selectedIndex || 0;
        this.settings[name] = vocab[index];
      };
      const defaultValue = vocab[0] || '';
      return html`<select class="dropdown control" @change=${updateDropdown}
          .value=${defaultValue} ?disabled=${vocab.length < 2}>${
        vocab.map((option, optionIndex) =>
            html`<option value=${optionIndex}>${option}</option>`)
      }</select>`;
    } else if (controlType instanceof Scalar) {
      // Render a slider.
      const {step, min_val: minVal, max_val: maxVal} = controlType;

      const updateSettings = (e: Event) => {
        this.settings[name] = (e.target as HTMLInputElement).value;
      };

      return html`<lit-numeric-input class="slider-holder"
          min=${minVal} max=${maxVal} step=${step} value=${this.settings[name]}
          @change=${updateSettings}></lit-numeric-input>`;
    } else if (controlType instanceof BooleanLitType) {
      // Render a checkbox.
      const toggleVal = () => {this.settings[name] = !this.settings[name];};
      return html`<lit-checkbox ?checked=${!!this.settings[name]}
          @change=${toggleVal}></lit-checkbox>`;
    } else if (controlType instanceof Tokens) {
      // Render a text input box and split on commas.
      const value = (this.settings[name] as string) || '';
      const updateText = (e: Event) => {
        const input = e.target! as HTMLInputElement;
        this.settings[name] = input.value.split(',').map(val => val.trim());
      };
      return html`<input class="control" type="text" @input=${updateText}
          .value=${value} />`;
    } else {
      // Render a text input box.
      const value = (this.settings[name] as string) || '';
      const updateText = (e: Event) => {
        this.settings[name] = (e.target as HTMLInputElement).value;
      };
      return html`<input class="control" type="text" @input=${updateText}
          .value=${value} />`;
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-interpreter-controls': InterpreterControls;
  }
}
