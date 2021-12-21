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
import {customElement} from 'lit/decorators';
import { html} from 'lit';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap} from '../services/color_service';
import {CallConfig, LitName, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {findSpecKeys, isLitSubtype} from '../lib/utils';
import {FocusData, FocusService} from '../services/focus_service';
import {AppState} from '../services/services';

import {styles} from './salience_map_module.css';
import {styles as sharedStyles} from '../lib/shared_styles.css';

/**
 * Results for calls to fetch salience.
 */
interface TokenSalienceResult {
  [key: string]: {tokens: string[], salience: number[]};
}

interface ImageSalienceResult {
  [key: string]: string;
}

interface FeatureSalienceMap {
  [feature: string]: number;
}

interface FeatureSalienceResult {
  [key: string]: {salience: FeatureSalienceMap};
}

/**
 * UI status for each interpreter.
 */
interface InterpreterState {
  salience: TokenSalienceResult|ImageSalienceResult|FeatureSalienceResult;
  autorun: boolean;
  isLoading: boolean;
  cmap: SalienceCmap;
  config?: CallConfig;
}

/**
 * A LIT module that displays gradient attribution scores for each token in the
 * selected inputs.
 */
@customElement('salience-map-module')
export class SalienceMapModule extends LitModule {
  static override title = 'Salience Maps';
  static override numCols = 6;
  static override duplicateForExampleComparison = true;
  static override template = (model = '', selectionServiceIndex = 0) => {
    return html`<salience-map-module model=${model} selectionServiceIndex=${
        selectionServiceIndex}></salience-map-module>`;
  };

  private readonly focusService = app.getService(FocusService);

  // Types that contain salience information to display.
  private static readonly salienceTypes: LitName[] =
      ['TokenSalience', 'ImageSalience', 'FeatureSalience'];

  static override get styles() {
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
  private state: {[name: string]: InterpreterState} = {};

  shouldRunInterpreter(name: string) {
    return this.state[name].autorun;
  }

  override firstUpdated() {
    const interpreters = this.appState.metadata.interpreters;
    const validInterpreters =
        this.appState.metadata.models[this.model].interpreters;
    const state: {[name: string]: InterpreterState} = {};
    for (const key of validInterpreters) {
      const salienceKeys = findSpecKeys(
          interpreters[key].metaSpec, SalienceMapModule.salienceTypes);
      if (salienceKeys.length === 0) {
        continue;
      }
      const salienceSpecInfo = interpreters[key].metaSpec[salienceKeys[0]];
      state[key] = {
        autorun: !!salienceSpecInfo.autorun,
        isLoading: false,
        salience: {},
        cmap: !!salienceSpecInfo.signed ?
            new SignedSalienceCmap(/* gamma */ 4.0) :
            new UnsignedSalienceCmap(/* gamma */ 4.0),
      };
    }
    this.state = state;

    this.reactImmediately(() => this.focusService.focusData, focusData => {
      this.handleFocus(this.focusService.focusData);
    });
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
        this.state[name].config, `Running ${name}`);
    const salience = await this.loadLatest(`interpretations-${name}`, promise);
    this.state[name].isLoading = false;
    if (salience === null) return;

    this.state[name].salience = salience[0];
  }

  handleFocus(focusData: FocusData|null) {
    this.shadowRoot!.querySelectorAll('.tokens-group').forEach((e) => {
      // For each token group we have a single tooltip,
      // which we reposition to the focused token.
      const tooltip = e.querySelector('.salience-tooltip') as HTMLElement;
      if (focusData == null || focusData.fieldName == null) {
        tooltip.style.visibility = 'hidden';
      } else if (focusData.datapointId ===
                 this.selectionService.primarySelectedInputData!.id) {
        const tokens = e.querySelectorAll(
            `.salient-token[data-gradKey="${focusData.fieldName}"]`);
        tokens.forEach((t, i) => {
          const tokenElement = t as HTMLElement;
          if (i !== focusData.subField) {
            return;
          }
          tooltip.innerText = tokenElement.dataset['tooltip']!;
          tooltip.style.visibility = 'visible';
          const tokenBcr = tokenElement.getBoundingClientRect();
          const groupBcr = e.getBoundingClientRect();
          const tooltipLeft =
              ((tokenBcr.left + tokenBcr.right) / 2) - groupBcr.left;
          const tooltipTop = tokenBcr.bottom - groupBcr.top;
          tooltip.style.left = `${tooltipLeft}px`;
          tooltip.style.top = `${tooltipTop}px`;
        });
      }
    });
  }

  override updated() {
    super.updated();

    // Imperative tooltip implementation
    this.shadowRoot!.querySelectorAll('.tokens-group').forEach((e) => {
      // For each token group we have a single tooltip,
      // which we reposition to the current element on mouseover.
      const tokens = e.querySelectorAll('.salient-token');
      const self = this;
      tokens.forEach((t, i) => {
        (t as HTMLElement).onmouseover = function() {
          self.focusService.setFocusedField(
              self.selectionService.primarySelectedInputData!.id,
              'input',
              (this as HTMLElement).dataset['gradkey']!,
              i);
        };
        // tslint:disable-next-line:only-arrow-functions
        (t as HTMLElement).onmouseout = function() {
          self.focusService.clearFocus();
        };
      });
    });
  }

  renderToken(token: string, salience: number, cmap: SalienceCmap, gradKey: string) {
    const tokenStyle = styleMap({
      'color': cmap.textCmap(salience),
      'background-color': cmap.bgCmap(salience)
    });

    if (gradKey != null) {
      return html`
        <div class="salient-token" style=${tokenStyle}
          data-tooltip=${salience.toPrecision(3)} data-gradkey=${gradKey}>
          ${token}
        </div>`;
    } else {
      return html`
        <div class="salient-token" style=${tokenStyle}
          data-tooltip=${salience.toPrecision(3)}>
          ${token}
        </div>`;
    }
  }

  renderImage(
      salience: ImageSalienceResult, gradKey: string) {
    const salienceImage = salience[gradKey];
    return html`<img src='${salienceImage}'></img>`;
  }

  // TODO(lit-dev): consider moving this to a standalone viz class.
  renderTokens(salience: TokenSalienceResult, gradKey: string,
               cmap: SalienceCmap) {
    const tokens = salience[gradKey].tokens;
    const saliences = salience[gradKey].salience;
    const tokensDOM = tokens.map(
        (token: string, i: number) =>
            this.renderToken(token, saliences[i], cmap, gradKey));

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

  renderFeatureSalience(salience: FeatureSalienceResult, gradKey: string,
                        cmap: SalienceCmap) {
    const saliences = salience[gradKey].salience;
    const features = Object.keys(saliences).sort(
        (a, b) => saliences[b] - saliences[a]);
    const tokensDOM = features.map(
        (feat: string) => {
          const val =
              this.selectionService.primarySelectedInputData!.data[feat];
          const str = `${feat}: ${val}`;
          return this.renderToken(str, saliences[feat], cmap, gradKey);
    });

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

  renderGroup(
      salience: TokenSalienceResult|ImageSalienceResult|FeatureSalienceResult,
      gradKey: string, cmap: SalienceCmap) {
    const spec = this.appState.getModelSpec(this.model);
    if (isLitSubtype(spec.output[gradKey], 'ImageGradients')) {
      salience = salience as ImageSalienceResult;
      return this.renderImage(salience, gradKey);
    } else if (isLitSubtype(spec.output[gradKey], 'FeatureSalience')) {
      salience = salience as FeatureSalienceResult;
      return this.renderFeatureSalience(salience, gradKey, cmap);
    } else {
      salience = salience as TokenSalienceResult;
      return this.renderTokens(salience, gradKey, cmap);
    }
  }

  renderSpinner() {
    return html`
      <div class="spinner-container">
        <lit-spinner size=${24} color="var(--app-secondary-color)">
        </lit-spinner>
      </div>
    `;
  }

  renderControls() {
    return Object.keys(this.state).map(name => {
      const toggleAutorun = () => {
        this.state[name].autorun = !this.state[name].autorun;
      };
      // clang-format off
      return html`
        <lit-checkbox label=${name}
         ?checked=${this.state[name].autorun}
         @change=${toggleAutorun}>
        </lit-checkbox>
      `;
      // clang-format on
    });
  }

  renderTable() {
    const controlsApplyCallback = (event: Event) => {
      // tslint:disable-next-line:no-any
      const name =  (event as any).detail.name;
      // tslint:disable-next-line:no-any
      this.state[name].config = (event as any).detail.settings;
      this.runInterpreter(name);
    };

    // clang-format off
    const renderMethodControls = (name: string) => {
      const spec = this.appState.metadata.interpreters[name].configSpec;
      const clonedSpec = JSON.parse(JSON.stringify(spec)) as Spec;
      for (const fieldName of Object.keys(clonedSpec)) {
        // If the generator uses a field matcher, then get the matching
        // field names from the specified spec and use them as the vocab.
        if (isLitSubtype(clonedSpec[fieldName],
                         ['FieldMatcher', 'MultiFieldMatcher'])) {
          clonedSpec[fieldName].vocab =
              this.appState.getSpecKeysFromFieldMatcher(
                  clonedSpec[fieldName], this.model);
        }
      }
      return html`
          <lit-interpreter-controls
            .spec=${clonedSpec}
            .name=${name}
            @interpreter-click=${controlsApplyCallback}>
          </lit-interpreter-controls>`;
    };
    return html`
      <table>
        ${Object.keys(this.state).map(name => {
          if (!this.state[name].autorun) {
            return null;
          }
          const salience = this.state[name].salience;
          const description =
              this.appState.metadata.interpreters[name].description || name;
          return html`
            <tr class='method-row'>
              <th class='group-label' title=${description}>
                ${renderMethodControls(name)}
              </th>
              <td class=${classMap({'group-container': true,
                                    'loading': this.state[name].isLoading})}>
                ${Object.keys(salience).map(gradKey =>
                  this.renderGroup(salience, gradKey, this.state[name].cmap))}
                ${this.state[name].isLoading ? this.renderSpinner() : null}
              </td>
            </tr>
          `;
        })}
      </table>
    `;
    // clang-format on
  }

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        <div class='module-toolbar'>
          ${this.renderControls()}
        </div>
        <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
          ${this.renderTable()}
        </div>
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    // TODO(b/204779018): Add appState interpreters to method arguments.

    // Ensure there are salience interpreters for loaded models.
    const appState = app.getService(AppState);
    for (const modelInfo of Object.values(modelSpecs)) {
      for (let i = 0; i < modelInfo.interpreters.length; i++) {
        const interpreterName = modelInfo.interpreters[i];
        if (appState.metadata == null) {
          return false;
        }
        const interpreter = appState.metadata.interpreters[interpreterName];
        const salienceKeys = findSpecKeys(
            interpreter.metaSpec, SalienceMapModule.salienceTypes);
        if (salienceKeys.length !== 0) {
          return true;
        }
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
