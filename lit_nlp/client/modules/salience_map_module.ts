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
 * LIT module for salience maps, such as gradients or LIME.
 */

import '../elements/checkbox';
import '../elements/spinner';

// tslint:disable:no-new-decorators
import {customElement, html, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {ModelsMap, Spec} from '../lib/types';
import {findSpecKeys} from '../lib/utils';

import {styles} from './salience_map_module.css';
import {styles as sharedStyles} from './shared_styles.css';

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
  canRun?: (modelSpec: Spec) => boolean;
}

abstract class SalienceCmap {
  // Exponent for computing luminance values from salience scores.
  // A higher value gives higher contrast for small (close to 0) salience
  // scores.
  // See https://en.wikipedia.org/wiki/Gamma_correction
  constructor(protected gamma: number = 1.0) {}

  abstract bgCmap(d: number): string;
  abstract textCmap(d: number): string;
}

/**
 * Color map for unsigned (positive) salience maps.
 */
export class UnsignedSalienceCmap extends SalienceCmap {
  /**
   * Color mapper. Higher salience values get darker colors.
   */
  bgCmap(d: number) {
    const hue = 270;  // purple
    const intensity = (1 - d) ** this.gamma;
    return `hsl(${hue}, 50%, ${100 * intensity}%)`;
  }

  /**
   * Make sure tokens are legible when colormap is dark.
   */
  textCmap(d: number) {
    return (d > 0.1) ? 'white' : 'black';
  }
}

/**
 * Color map for signed salience maps.
 */
export class SignedSalienceCmap extends SalienceCmap {
  /**
   * Color mapper. Higher salience values get darker colors.
   */
  bgCmap(d: number) {
    let hue = 188;  // teal from WHAM
    if (d < 0.) {
      hue = 354;    // red(ish) from WHAM
      d = Math.abs(d);  // make positive for intensity
    }
    const intensity = (1 - d) ** this.gamma;
    return `hsl(${hue}, 50%, ${25 + 75 * intensity}%)`;
  }

  /**
   * Make sure tokens are legible when colormap is dark.
   */
  textCmap(d: number) {
    return (Math.abs(d) > 0.1) ? 'white' : 'black';
  }
}

/**
 * A LIT module that displays gradient attribution scores for each token in the
 * selected inputs.
 */
@customElement('salience-map-module')
export class SalienceMapModule extends LitModule {
  static title = 'Salience Maps';
  static numCols = 6;
  static duplicateForExampleComparison = true;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<salience-map-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></salience-map-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  // TODO: We may want the keys to be configurable through the UI at some point,
  // but for now they are constants.
  // TODO(lit-dev): consider making each interpreter a sub-module,
  // rather than multiplexing them here.
  // Downside is that would make it much harder to implement comparisons
  // between different salience methods, or to change the table arrangement
  // (e.g. group by input field, rather than salience technique).
  @observable
  private readonly state: {[name: string]: InterpreterState} = {
    'grad_norm': {
      autorun: true,
      isLoading: false,
      salience: {},
      cmap: new UnsignedSalienceCmap(/* gamma */ 4.0),
      // TODO(lit-dev): Should also check that this aligns with a token field,
      // here and in checkModule. Perhaps move component compatibility somewhere
      // central.
      canRun: (modelSpec: Spec) =>
          findSpecKeys(modelSpec, 'TokenGradients').length > 0
    },
    'grad_dot_input': {
      autorun: true,
      isLoading: false,
      salience: {},
      cmap: new SignedSalienceCmap(/* gamma */ 4.0),
      // TODO(lit-dev): Should also check that this aligns with a token field,
      // here and in checkModule. Perhaps move component compatibility somewhere
      // central.
      canRun: (modelSpec: Spec) =>
          findSpecKeys(modelSpec, 'TokenGradients').length > 0 &&
          findSpecKeys(modelSpec, 'TokenEmbeddings').length > 0
    },
    'integrated gradients': {
      autorun: false,
      isLoading: false,
      salience: {},
      cmap: new SignedSalienceCmap(/* gamma */ 4.0),
      // TODO(lit-dev): Should also check that this aligns with a token field,
      // here and in checkModule. Perhaps move component compatibility somewhere
      // central.
      canRun: (modelSpec: Spec) =>
          findSpecKeys(modelSpec, 'TokenGradients').length > 0 &&
          findSpecKeys(modelSpec, 'TokenEmbeddings').length > 0
    },
    'lime': {
      autorun: false,
      isLoading: false,
      salience: {},
      cmap: new SignedSalienceCmap(/* gamma */ 4.0)
    },
  };

  shouldRunInterpreter(name: string) {
    return this.state[name].autorun &&
        (this.state[name].canRun == null ||
         this.state[name].canRun!
         (this.appState.getModelSpec(this.model).output));
  }

  firstUpdated() {
    for (const name of Object.keys(this.state)) {
      // React to change in primary selection.
      const getData = () => this.selectionService.primarySelectedInputData;
      this.react(getData, data => {
        if (this.shouldRunInterpreter(name)) {
          this.runInterpreter(name);
        } else {
          this.state[name].salience = {}; /* clear output */
        }
      });
      // React to change in 'autorun' checkbox.
      const getAutorun = () => this.state[name].autorun;
      this.react(getAutorun, autorun => {
        if (this.shouldRunInterpreter(name)) {
          this.runInterpreter(name);
        }
        // If disabled, do nothing until selection changes.
      });

      // Initial update, to avoid duplicate calls.
      if (this.shouldRunInterpreter(name)) {
        this.runInterpreter(name);
      }
    }
  }

  private async runInterpreter(name: string) {
    const input = this.selectionService.primarySelectedInputData;
    if (input == null) {
      this.state[name].salience = {}; /* clear output */
      return;
    }

    this.state[name].isLoading = true;
    const promise = this.apiService.getInterpretations(
        [input], this.model, this.appState.currentDataset, name,
        /* callConfig */ undefined, `Running ${name}`);
    const salience = await this.loadLatest(`interpretations-${name}`, promise);
    this.state[name].isLoading = false;
    if (salience === null) return;

    this.state[name].salience = salience[0];
  }

  updated() {
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

  render() {
    // clang-format off
    return html`
      <table>
        ${Object.keys(this.state).map(name => {
          if (this.state[name].canRun != null &&
              !this.state[name].canRun!(this.appState.getModelSpec(
                  this.model).output)) {
            return null;
          }
          const salience = this.state[name].salience;
          const cmap = this.state[name].cmap;
          return html`
            <tr>
              <th class='group-label'>
                ${name}
                <lit-checkbox label="autorun"
                  ?checked=${this.state[name].autorun}
                  @change=${() => { this.state[name].autorun = !this.state[name].autorun;}}
                ></lit-checkbox>
              </th>
              <td class=${classMap({'group-container': true,
                                    'loading': this.state[name].isLoading})}>
                ${Object.keys(salience).map(gradKey =>
                  this.renderGroup(salience, gradKey, cmap))}
                ${this.state[name].isLoading ? this.renderSpinner() : null}
              </td>
            </tr>
          `;
        })}
      </table>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    for (const model of Object.keys(modelSpecs)) {
      const inputSpec = modelSpecs[model].spec.input;
      const outputSpec = modelSpecs[model].spec.output;
      const supportsGrads = (findSpecKeys(outputSpec, 'TokenGradients').length);
      const supportsLime = (findSpecKeys(inputSpec, 'TextSegment').length) &&
          ((findSpecKeys(outputSpec, 'MulticlassPreds').length) ||
           (findSpecKeys(outputSpec, 'RegressionScore').length));
      if (supportsGrads || supportsLime) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'salience-map-module': SalienceMapModule;
  }
}
