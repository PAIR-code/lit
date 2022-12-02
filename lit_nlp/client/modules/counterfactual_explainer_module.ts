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
 * LIT module for counterfactuals-based salience display.
 */

import '../elements/checkbox';
import '../elements/spinner';

// tslint:disable:no-new-decorators
import {property} from 'lit/decorators';
import {customElement} from 'lit/decorators';
import {css, html} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {TextSegment, MulticlassPreds} from '../lib/lit_types';
import {ModelInfoMap, Spec} from '../lib/types';
import {findSpecKeys, range} from '../lib/utils';
import {SalienceCmap, SignedSalienceCmap} from '../services/color_service';

import {styles as salienceMapStyles} from './salience_map_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

const COUNTERFACTUAL_INTERPRETER_NAME = 'counterfactual explainer';

/**
 * Results for calls to fetch salience.
 */
interface SalienceResult {
  [key: string]: {tokens: string[], salience: number[]};
}

/**
 * UI status for each interpreter.
 */
interface InterpreterState {
  salience: SalienceResult;
  autorun: boolean;
  isLoading: boolean;
  cmap: SalienceCmap;
}

/**
 * A LIT module that displays counterfactual-based attribution scores for each
 * token in the selected input.
 */
@customElement('counterfactual-explainer-module')
export class CounterfactualExplainerModule extends LitModule {
  static override title = 'Counterfactual Explanation';
  static override numCols = 10;
  static override collapseByDefault = true;
  static override duplicateForExampleComparison = true;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
  <counterfactual-explainer-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </counterfactual-explainer-module>`;
      };

  @property({type: String}) override model = '';
  @property({type: Number}) override selectionServiceIndex = 0;

  static override get styles() {
    return [
      sharedStyles,
      salienceMapStyles,
      css`
        .options-dropdown {
          margin-right: 10px;
          margin-top: 10px;
          margin-bottom: 10px;
          min-width: 100px;
      }
      `,
    ];
  }

  @observable selectedClass = 0;
  @observable selectedPredKey = '';

  // TODO: We may want the keys to be configurable through the UI at some point,
  // but for now they are constants.
  // TODO(lit-dev): consider making each interpreter a sub-module,
  // rather than multiplexing them here.
  // Downside is that would make it much harder to implement comparisons
  // between different salience methods, or to change the table arrangement
  // (e.g. group by input field, rather than salience technique).
  @observable
  private readonly state: InterpreterState = {
    autorun: false,
    isLoading: false,
    salience: {},
    cmap: new SignedSalienceCmap(/* gamma */ 4.0)
  };

  override firstUpdated() {
    // React to change in primary selection.
    const getPrimaryData = () => this.selectionService.primarySelectedInputData;
    this.react(getPrimaryData, data => {
      const autorun = this.state.autorun;
      if (autorun) {
        this.runInterpreter();
      } else {
        this.state.salience = {}; /* clear output */
      }
    });
    // React to change in overall selection if the selected input data is
    // passed for this interpreter.
    const getData = () => this.selectionService.selectedInputData;
    this.react(getData, data => {
      const autorun = this.state.autorun;
      if (autorun) {
        this.runInterpreter();
      } else {
        this.state.salience = {}; /* clear output */
      }
    });
    // React to change in 'autorun' checkbox.
    const getAutorun = () => this.state.autorun;
    this.react(getAutorun, autorun => {
      if (autorun) {
        this.runInterpreter();
      }
      // If disabled, do nothing until selection changes.
    });

    // React to change in 'Class to explain' dropdown.
    const getSelectedClass = () => this.selectedClass;
    this.react(getSelectedClass, selectedClass => {
      const autorun = this.state.autorun;
      if (autorun) {
        this.runInterpreter();
      }
    });

    // React to change in 'Model head' dropdown.
    const getSelectedPredKey = () => this.selectedPredKey;
    this.react(getSelectedPredKey, selectedPredKey => {
      const autorun = this.state.autorun;
      if (autorun) {
        this.runInterpreter();
      }
    });

    // Initial update, to avoid duplicate calls.
    if (this.state.autorun) {
      this.runInterpreter();
    }
  }

  private async runInterpreter() {
    const name = COUNTERFACTUAL_INTERPRETER_NAME;
    const primaryInput = this.selectionService.primarySelectedInputData;
    if (primaryInput == null) {
      this.state.salience = {}; /* clear output */
      return;
    }

    const counterfactuals = this.selectionService.selectedInputData.filter(
        (selectedInputData) => selectedInputData.id !== primaryInput.id);

    // Return if the selected pred key hasn't been set.
    if (this.selectedPredKey === '') return;
    const config = {
      'class_to_explain': this.selectedClass,
      'lowercase_tokens': true,
      'pred_key': this.selectedPredKey
    };

    this.state.isLoading = true;
    const promise = this.apiService.getInterpretations(
        [primaryInput, ...counterfactuals], this.model,
        this.appState.currentDataset, name, config, `Running ${name}`);
    const salience = await this.loadLatest(`interpretations-${name}`, promise);
    this.state.isLoading = false;
    if (salience === null) return;

    this.state.salience = salience[0];
  }

  override updated() {
    super.updated();

    // Imperative tooltip implementation
    this.shadowRoot!.querySelectorAll('.tokens-group').forEach((e) => {
      // For each token group we have a single tooltip,
      // which we reposition to the current element on mouseover.
      const tooltip = e.querySelector('.salience-tooltip') as HTMLElement;
      const tokens = e.querySelectorAll('.salient-token');
      tokens.forEach((t) => {
        (t as HTMLElement).onmouseover = function() {
          tooltip.innerText = (this as HTMLElement).dataset['tooltip']!;
          tooltip.style.visibility = 'visible';
          const bcr = (this as HTMLElement).getBoundingClientRect();
          tooltip.style.left = `${(bcr.left + bcr.right) / 2}px`;
          tooltip.style.top = `${bcr.bottom}px`;
        };
        // tslint:disable-next-line:only-arrow-functions
        (t as HTMLElement).onmouseout = function() {
          tooltip.style.visibility = 'hidden';
        };
      });
    });
  }

  renderToken(token: string, salience: number, cmap: SalienceCmap) {
    const tokenStyle = styleMap({
      'color': cmap.textCmap(salience),
      'background-color': cmap.bgCmap(salience)
    });

    return html`
      <div class="salient-token" style=${tokenStyle}
        data-tooltip=${salience.toPrecision(3)}>
        ${token}
      </div>`;
  }

  // TODO(lit-dev): consider moving this to a standalone viz class.
  renderGroup(salience: SalienceResult, gradKey: string, cmap: SalienceCmap) {
    const tokens = salience[gradKey].tokens;
    const saliences = salience[gradKey].salience;

    const tokensDOM = tokens.map(
        (token: string, i: number) =>
            this.renderToken(token, saliences[i], cmap));

    // clang-format off
    return html`
      <div class="tokens-group">
        <div class="tokens-group-title">
          ${gradKey}
        </div>
        <div class="tokens-holder">
          ${tokensDOM}
        </div>
        <div class="salience-tooltip">
        </div>
      </div>
    `;
    // clang-format on
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${24} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }

  override renderImpl() {
    const salience = this.state.salience;
    const cmap = this.state.cmap;

    // Get a list of prediction keys and the number of classes for the selected
    // prediction key to be displayed as dropdown options.
    const outputSpec = this.appState.currentModelSpecs[this.model].spec.output;
    const predKeys = findSpecKeys(outputSpec, MulticlassPreds);
    if (!predKeys.length) {
      return;
    }

    if (!predKeys.includes(this.selectedPredKey)) {
      this.selectedPredKey = predKeys[0];
    }

    const predictionLabels =
        (outputSpec[this.selectedPredKey] as MulticlassPreds).vocab;
    const classes: number[] =
        (predictionLabels == null) ? [] : range(predictionLabels.length);
    if (!classes.includes(this.selectedClass)) {
      this.selectedClass = classes[classes.length - 1];
    }

    const predKeyOptions = predKeys.map((option, optionIndex) => {
      return html`<option value=${optionIndex}>
              ${option}</option>`;
    });
    const classOptions = classes.map((option, optionIndex) => {
      return html`<option value=${optionIndex} >
              ${option}</option>`;
    });

    const classChanged = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      if (select != null) {
        this.selectedClass = +select.options[select.selectedIndex].value;
      }
    };
    const predKeyChanged = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      if (select != null) {
        this.selectedPredKey = select.options[select.selectedIndex].value;
      }
    };

    // clang-format off
    return html`
      <div class='config'>
        <label class="dropdown-label">Class to explain:</label>
        <select class="dropdown options-dropdown" @change=${classChanged}>
          ${classOptions}
        </select>
        <label class="dropdown-label">Model head:</label>
        <select class="dropdown options-dropdown" @change=${predKeyChanged}>
          ${predKeyOptions}
        </select>
        <div class='group-label'>
          <lit-checkbox label="autorun"
            ?checked=${this.state.autorun}
            @change=${() => { this.state.autorun = !this.state.autorun;}}
          ></lit-checkbox>
        </div>
      </div>
      <table>
        <tr>
          <td class=${classMap({'group-container': true,
                                'loading': this.state.isLoading})}>
            ${Object.keys(salience).map(gradKey =>
              this.renderGroup(salience, gradKey, cmap))}
            ${this.state.isLoading ? this.renderSpinner() : null}
          </td>
        </tr>
      </table>
    `;
    // clang-format on
  }

  // tslint:disable-next-line:no-any
  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    for (const model of Object.keys(modelSpecs)) {
      const inputSpec = modelSpecs[model].spec.input;
      const outputSpec = modelSpecs[model].spec.output;

      const supportsLemon = (findSpecKeys(inputSpec, TextSegment).length) &&
          (findSpecKeys(outputSpec, MulticlassPreds).length);
      if (supportsLemon) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'counterfactual-explainer-module': CounterfactualExplainerModule;
  }
}
