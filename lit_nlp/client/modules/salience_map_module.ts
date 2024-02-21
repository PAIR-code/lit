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

import '../elements/spinner';
import '../elements/numeric_input';
import '../elements/popup_container';
import '../elements/token_chips';

import {html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators.js';
import {styleMap} from 'lit/directives/style-map.js';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {InterpreterClick} from '../elements/interpreter_controls';
import {TokenWithWeight} from '../elements/token_chips';
import {FeatureSalience, FieldMatcher, ImageGradients, ImageSalience, LitTypeTypesList, LitTypeWithParent, MulticlassPreds, RegressionScore, Salience, SalienceTargetInfo, TokenGradients, TokenSalience, FrameSalience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {CallConfig, type IndexedInput, ModelInfoMap, type Preds, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {argmax, cloneSpec, findSpecKeys, makeModifiedInput} from '../lib/utils';
import {SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap} from '../services/color_service';
import {FocusService} from '../services/focus_service';
import {AppState} from '../services/services';

import {styles} from './salience_map_module.css';

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

/**
 * Results for calls to fetch salience for features.
 */
export interface FeatureSalienceResult {
  [key: string]: {salience: FeatureSalienceMap};
}

/** Different types of salience results supported in this module. */
export type SalienceResult = TokenSalienceResult | ImageSalienceResult |
                      FeatureSalienceResult;

const SUPPORTED_SALIENCE_TYPES = [TokenSalience, FeatureSalience, ImageSalience, FrameSalience];

const TARGET_SELECTOR_SUPPORTED_TYPES: LitTypeTypesList =
    [MulticlassPreds, RegressionScore];
type TargetFieldType = MulticlassPreds|RegressionScore;

/**
 * UI status for each interpreter.
 */
interface InterpreterState {
  salience: SalienceResult;
  autorun: boolean;
  isLoading: boolean;
  cmap: SalienceCmap;
  config?: CallConfig;
}

const LEGEND_INFO_TITLE_SIGNED =
    "Salience is relative to the model's prediction of a class. A positive " +
    "score (more green) for a token means that token influenced the model to " +
    "predict that class.";

const LEGEND_INFO_TITLE_UNSIGNED =
    "Salience is relative to the model's prediction of a class. A larger " +
    "score (more purple) for a token means that token was more influential " +
    "on the model's prediction of that class.";

/**
 * A LIT module that displays gradient attribution scores for each token in the
 * selected inputs.
 */
@customElement('salience-map-module')
export class SalienceMapModule extends LitModule {
  static override title = 'Salience Maps';
  static override infoMarkdown =
      `Input salience methods try to explain model predictions as a heatmap
      over input features, such as tokens.<br>
      [Learn more.](https://pair-code.github.io/lit/documentation/components.md#token-based-salience)`;
  static override numCols = 6;
  static override duplicateForExampleComparison = true;
  static override template = (model: string, selectionServiceIndex: number, shouldReact: number) => {
    return html`<salience-map-module model=${model} selectionServiceIndex=${
        selectionServiceIndex} .shouldReact=${shouldReact}></salience-map-module>`;
  };

  private readonly focusService = app.getService(FocusService);
  private isRenderImage = false;
  private hasSignedSalience = false;
  private hasUnsignedSalience = false;

  static override get styles() {
    return [sharedStyles, styles];
  }

  // For color legend
  @observable private cmapGamma = 2.0;
  @computed
  get signedCmap() {
      return new SignedSalienceCmap(/* gamma */ this.cmapGamma);
  }
  @computed
  get unsignedCmap() {
      return new UnsignedSalienceCmap(/* gamma */ this.cmapGamma);
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

  @observable private currentData?: IndexedInput;
  @observable private currentPreds?: Preds;
  // Index into possible labels.
  @observable private targetField?: string;
  @observable private salienceTarget?: number;

  // Check that the target selector is available for gradient-based salience.
  // TODO(b/205996131): remove this and always show dropdown, once everything
  // is updated such that this accurately reflects the class being explained.
  @computed
  get compatibleWithTargetSelector(): boolean {
      const {input: inputSpec, output: outputSpec} =
          this.appState.getModelSpec(this.model);
      const gradKeys = findSpecKeys(outputSpec, TokenGradients);
      // Input spec fields
      const parentFields =
          this.targetFields
              .map(fname => (outputSpec[fname] as TargetFieldType).parent)
              .filter(pname => pname != null);
      for (const k of gradKeys) {
        const gradTargetKey =
            (outputSpec[k] as TokenGradients).grad_target_field_key;
        if (gradTargetKey == null) continue;
        // Check that this field is in the model's input spec.
        if (inputSpec[gradTargetKey] == null) return false;
        // Also check that this is a label field, for some output.
        if (parentFields.indexOf(gradTargetKey) === -1) return false;
      }
      return true;
  }

  @computed
  get targetFields(): string[] {
      const outputSpec = this.appState.getModelSpec(this.model).output;
      return findSpecKeys(outputSpec, TARGET_SELECTOR_SUPPORTED_TYPES);
  }

  @computed
  get targetFieldSpec(): TargetFieldType|null {
      if (this.targetField == null) return null;

      // this.targetField is selected from options in this.targetFields,
      // so if not undefined it will always be found in the output spec.
      return (
          this.appState.getModelSpec(this.model).output[this.targetField] as
          TargetFieldType);
  }

  @computed
  get inputLabelField(): string|null {
      if (this.targetFieldSpec == null) return null;

      return (this.targetFieldSpec as LitTypeWithParent).parent ?? null;
  }

  @computed
  get labelVocab(): string[] {
      if (this.targetFieldSpec == null) return [];

      if (this.targetFieldSpec instanceof MulticlassPreds) {
        return this.targetFieldSpec.vocab;
      }
      // Regression tasks, or unrecognized.
      return [];
  }

  @computed
  get modifiedInput(): IndexedInput|null {
      const input = this.currentData;
      if (input == null) return null;

      // Don't modify input if there's no target label.
      if (this.inputLabelField == null || this.salienceTarget === undefined) {
        return input;
      }

      // If newLabel matches the existing value, this will return input
      // unchanged.
      const newLabel = this.labelVocab[this.salienceTarget];
      return makeModifiedInput(
          input, {[this.inputLabelField]: newLabel}, 'SalienceMapModule');
  }

  shouldRunInterpreter(name: string) {
    return this.state[name].autorun;
  }

  private clearSalience() {
    for (const key of Object.keys(this.state)) {
      this.state[key].salience = {};
    }
  }

  override firstUpdated() {
    const interpreters = this.appState.metadata.interpreters;
    const validInterpreters =
        this.appState.metadata.models[this.model].interpreters;
    const state: {[name: string]: InterpreterState} = {};
    for (const key of validInterpreters) {
      const salienceKeys =
          findSpecKeys(interpreters[key].metaSpec, SUPPORTED_SALIENCE_TYPES);
      if (salienceKeys.length === 0) {
        continue;
      }
      const salienceSpecInfo =
          interpreters[key].metaSpec[salienceKeys[0]] as Salience;

      if (salienceSpecInfo instanceof ImageSalience) {
        this.isRenderImage = true;
      }

      // determine whether we need to show signedSalience or unSignedSalience
      this.hasSignedSalience = this.hasSignedSalience ||
          !!salienceSpecInfo.signed;
      this.hasUnsignedSalience = this.hasUnsignedSalience ||
          !(!!salienceSpecInfo.signed);

      state[key] = {
        autorun: !!salienceSpecInfo.autorun,
        isLoading: false,
        salience: {},
        cmap: !!salienceSpecInfo.signed ?
            this.signedCmap :
            this.unsignedCmap,
      };
    }
    this.state = state;

    this.reactImmediately(() => this.targetFields, targetFields => {
      if (targetFields.length <= 0) {
        // Two ways this can happen:
        // - If model is compatible with salience methods but doesn't use
        //   one of the compatible target field types, then we just don't show
        //   the target selector.
        // - If model is not compatible but this module is still shown; see
        //   b/205309955.
        console.warn(
            `Salience Map module: model "${this.model}" has no ` +
            'compatible output fields. Salience methods may still work, ' +
            'but you will not be able to change the target from the UI.');
        this.targetField = undefined;
      }
      this.targetField = targetFields[0];
    });

    // React to change in primary selection.
    this.reactImmediately(
        () => this.selectionService.primarySelectedInputData, data => {
          this.updateToSelection(data);
          // Also run any applicable interpreters.
          for (const name of Object.keys(this.state)) {
            if (this.shouldRunInterpreter(name)) {
              this.runInterpreter(name);
            }
          }
        });

    for (const name of Object.keys(this.state)) {
      this.react(() => this.salienceTarget, () => {
        if (this.shouldRunInterpreter(name)) {
          this.runInterpreter(name);
        } else {
          this.state[name].salience = {}; /* clear output */
        }
      });
      // React to change in 'autorun' checkbox.
      const getAutorun = () => this.state[name].autorun;
      this.react(getAutorun, () => {
        if (this.shouldRunInterpreter(name)) {
          this.runInterpreter(name);
        }
        // If disabled, do nothing until selection changes.
      });
    }
  }

  private async updateToSelection(input: IndexedInput|null) {
    this.currentData = undefined;
    this.currentPreds = undefined;
    this.salienceTarget = undefined;
    this.clearSalience();

    if (input == null) return;

    // Update data before waiting for backend call.
    this.currentData = input;

    // Fetch predictions
    const promise = this.apiService.getPreds(
        [input], this.model, this.appState.currentDataset,
        TARGET_SELECTOR_SUPPORTED_TYPES, [],
        'Getting targets from model prediction');
    const results = await this.loadLatest('modelPreds', promise);
    if (results === null) return;

    // Update data again, in case selection changed rapidly.
    this.currentData = input;
    this.currentPreds = results[0];

    // Set salience target to index of model's argmax prediction.
    if (this.targetFieldSpec instanceof MulticlassPreds) {
      const preds = this.currentPreds[this.targetField!];
      this.salienceTarget = argmax(preds);
    } else {
      this.salienceTarget = undefined;
    }
  }

  private async runInterpreter(name: string) {
    const input = this.modifiedInput;
    if (input == null) {
      this.state[name].salience = {}; /* clear output */
      return;
    }

    let config = this.state[name].config;
    const configSpec = this.appState.metadata.interpreters[name].configSpec;
    for (const key of findSpecKeys(configSpec, [SalienceTargetInfo])) {
      config = Object.assign(
          {}, config,
          {[key]: {field: this.targetField, index: this.salienceTarget}});
    }
    // If there is no SalienceTargetInfo in the spec, config should still be
    // null if no fields are set.

    this.state[name].isLoading = true;
    const promise = this.apiService.getInterpretations(
        [input], this.model, this.appState.currentDataset, name, config,
        `Running ${name}`);
    const salience = await this.loadLatest(`interpretations-${name}`, promise);
    this.state[name].isLoading = false;
    if (salience === null) return;

    this.state[name].salience = salience[0];
  }

  renderImageSalience(salience: ImageSalienceResult, gradKey: string) {
    const salienceImage = salience[gradKey];
    return html`<img src='${salienceImage}'></img>`;
  }

  private renderHeaderControls() {
    if (!this.compatibleWithTargetSelector) return null;

    // Target selector incompatible.
    if (this.targetFields.length <= 0) return null;

    const onChangeField = (e: Event) => {
      this.targetField = (e.target as HTMLSelectElement).value;
    };

    const onChangeTarget = (e: Event) => {
      this.salienceTarget = (e.target as HTMLSelectElement).selectedIndex;
    };

    const renderClassSelector = () => {
      // clang-format off
      return html`
        <label class="dropdown-label">Class to explain:</label>
        <select class="dropdown salience-target-dropdown"
          @change=${onChangeTarget}
          .value=${this.labelVocab[this.salienceTarget!]}>
          ${this.labelVocab.map((target, i) =>
            html`<option value=${target}
                    ?selected=${i === this.salienceTarget}>
                  ${target}
                  </option>`)}
        </select>
      `;
      // clang-format on
    };

    // clang-format off
    return html`
      <div class="module-toolbar">
        <div class="controls-group">
          <label class="dropdown-label">Target field:</label>
          <select class="dropdown salience-target-dropdown"
            @change=${onChangeField}>
            ${this.targetFields.map(fieldName =>
              html`<option value=${fieldName}
                     ?selected=${fieldName === this.targetField}>
                    ${fieldName}
                   </option>`)}
          </select>
          ${this.labelVocab.length > 0 ? renderClassSelector() : null}
        </div>
        <div class="controls-group">
        </div>
      </div>`;
    // clang-format on
  }

  renderColorLegend(
      legendLabel: string, colorMap: SalienceCmap, numBlocks: number) {
    const tooltipText = legendLabel === 'Signed' ? LEGEND_INFO_TITLE_SIGNED :
                                                 LEGEND_INFO_TITLE_UNSIGNED;

    // clang-format off
    return html`
      <color-legend legendType=${LegendType.SEQUENTIAL}
        label=${legendLabel}
        .paletteTooltipText=${tooltipText}
        .scale=${colorMap.asScale()}
        numBlocks=${numBlocks}>
      </color-legend>`;
    // clang-format on
  }

  static readonly sliderTooltipText =
      'A larger gamma value makes lower salience tokens more visible.';

  // TODO(b/242164240): Refactor the code once we decide how to address
  // the contrast problem.
  renderSlider() {
    const onChangeGamma = (e: Event) => {
      this.cmapGamma = Number((e.target as HTMLInputElement).value);

      // Update the cmap values in state
      const interpreters = this.appState.metadata.interpreters;
      const validInterpreters =
        this.appState.metadata.models[this.model].interpreters;
      for (const key of validInterpreters) {
        const salienceKeys = findSpecKeys(interpreters[key].metaSpec, Salience);
        if (salienceKeys.length === 0) {
          continue;
        }
        const salienceSpecInfo =
          interpreters[key].metaSpec[salienceKeys[0]] as Salience;
        this.state[key].cmap = !!salienceSpecInfo.signed ?
          this.signedCmap :
          this.unsignedCmap;
      }
    };

    // clang-format off
    return html`
      <div class='slider-controls'>
        <lit-tooltip .tooltipPosition=${'above'}
          .content=${SalienceMapModule.sliderTooltipText}>
          <label for="gamma-slider" slot="tooltip-anchor">Gamma:</label>
        </lit-tooltip>
        <lit-numeric-input min="0.25" max="6" step="0.25"
          value="${this.cmapGamma}" @change=${onChangeGamma}>
        </lit-numeric-input>
      </div>`;
    // clang-format on
  }

  renderFooter() {
    // clang-format off
    return html`
      <div class="module-footer">
        ${this.hasSignedSalience
          ? this.renderColorLegend("Signed", this.signedCmap, 7)
          : null}
        ${this.hasUnsignedSalience
          ? this.renderColorLegend("Unsigned", this.unsignedCmap, 5)
          : null}
        ${this.hasSignedSalience || this.hasUnsignedSalience
          ? this.renderSlider() : null}
      </div>`;
    // clang-format on
  }

  renderTokenSalience(
      tokens: string[], scores: number[], gradKey: string, cmap: SalienceCmap,
      title?: string) {
    const tokensWithWeights: TokenWithWeight[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const onMouseover = () => {
        this.focusService.setFocusedField(
            this.selectionService.primarySelectedInputData!.id, 'input',
            gradKey, i);
      };
      const onMouseout = () => {
        this.focusService.clearFocus();
      };
      const focusData = this.focusService.focusData;
      const forceShowTooltip = focusData?.fieldName === gradKey &&
          focusData?.datapointId === this.selectionService.primarySelectedId &&
          focusData?.subField === i;
      tokensWithWeights.push({
        token: tokens[i],
        weight: scores[i],
        onMouseover,
        onMouseout,
        forceShowTooltip
      });
    }

    // clang-format off
    return html`
      <div class="tokens-group">
        ${title? html`<div class="tokens-group-title">${title}</div>` : null}
        <lit-token-chips
          .tokensWithWeights=${tokensWithWeights}
          .cmap=${cmap}>
        </lit-token-chips>
      </div>`;
    // clang-format on
  }

  renderFeatureSalience(
      salience: FeatureSalienceResult, gradKey: string, cmap: SalienceCmap,
      title?: string) {
    const saliences = salience[gradKey].salience;
    const features = Object.keys(saliences).sort(
        (a, b) => saliences[b] - saliences[a]);
    const data = this.selectionService.primarySelectedInputData!.data;
    const featureTexts =
        features.map((feat: string) => `${feat}: ${data[feat]}`);
    const values = features.map((feat: string) => saliences[feat]);

    return this.renderTokenSalience(featureTexts, values, gradKey, cmap, title);
  }

  renderGroup(
      salience: SalienceResult, spec: Spec, gradKey: string, cmap: SalienceCmap,
      title?: string) {
    if (spec[gradKey] instanceof ImageGradients) {
      salience = salience as ImageSalienceResult;
      return this.renderImageSalience(salience, gradKey);
    } else if (spec[gradKey] instanceof FeatureSalience) {
      salience = salience as FeatureSalienceResult;
      return this.renderFeatureSalience(salience, gradKey, cmap, title);
    } else {
      salience = salience as TokenSalienceResult;
      return this.renderTokenSalience(
          salience[gradKey].tokens, salience[gradKey].salience, gradKey, cmap,
          title);
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

  private controlsApplyCallback(event: CustomEvent<InterpreterClick>) {
    const name = event.detail.name;
    this.state[name].config = event.detail.settings;
    this.runInterpreter(name);
  }

  private renderMethodControls(name: string) {
    const spec = this.appState.metadata.interpreters[name].configSpec;
    // Don't render controls if there aren't any.
    if (Object.keys(spec).length === 0) return null;

    const clonedSpec = cloneSpec(spec);
    for (const [fieldName, fieldSpec] of Object.entries(clonedSpec)) {
      // Don't show interpreter controls for the special SalienceTargetInfo
      // type; this gets special handing in runInterpreter().
      if (fieldSpec instanceof SalienceTargetInfo) {
        delete clonedSpec[fieldName];
        continue;
      }

      // If the generator uses a field matcher, then get the matching
      // field names from the specified spec and use them as the vocab.
      if (fieldSpec instanceof FieldMatcher) {
        fieldSpec.vocab =
            this.appState.getSpecKeysFromFieldMatcher(fieldSpec, this.model);
      }
    }

    const description =
        this.appState.metadata.interpreters[name].description ?? name;
    const descriptionPreview = description.split('\n')[0];

    // Right-anchor the controls popup.
    const popupStyle = styleMap({'--popup-right': '0'});
    // The "control" slot in <popup-container> renders in-place and uses the
    // given template as a toggle control for the popup.
    // This allows control state to remain inside <popup-container>, freeing
    // us from having to keep track of another set of booleans here.
    // clang-format off
    return html`
      <popup-container style=${popupStyle}>
        <div class="controls-toggle" slot="toggle-anchor">
          <mwc-icon class='icon-button'>settings</mwc-icon>
        </div>
        <div class="controls-panel">
          <lit-interpreter-controls .spec=${clonedSpec} .name=${name}
                      .description=${descriptionPreview}
                      noexpand opened
                      @interpreter-click=${this.controlsApplyCallback}>
          </lit-interpreter-controls>
        </div>
      </popup-container>`;
    // clang-format on
  }

  renderMethodRow(name: string) {
    // TODO(b/217724273): figure out a more elegant way to handle
    // variable-named output fields with metaSpec.
    const {output} = this.appState.getModelSpec(this.model);
    const {metaSpec} = this.appState.metadata.interpreters[name];
    const spec = {...metaSpec, ...output};
    const salience = this.state[name].salience;
    const description =
        this.appState.metadata.interpreters[name].description ?? name;

    const toggleAutorun = () => {
      this.state[name].autorun = !this.state[name].autorun;
    };

    const onlyOneGroup = Object.keys(salience).length === 1;
    const salienceContent = Object.keys(salience).map(
        gradKey => this.renderGroup(
            salience, spec, gradKey, this.state[name].cmap,
            /* title */ onlyOneGroup ? undefined : gradKey));

    // The "bar-content" slot in <expansion-panel> renders inline between
    // the label and the expander toggle.
    // clang-format off
    return html`
      <div class='method-row' id='${name.toLowerCase()}'>
        <expansion-panel .label=${name} ?expanded=${this.state[name].autorun}
                         .description=${description}
                          @expansion-toggle=${toggleAutorun}>
          <div slot="bar-content">
            ${this.renderMethodControls(name)}
          </div>
          <div class='method-row-contents'>
            <div class='method-results'>
              ${this.selectionService.primarySelectedInputData != null ?
                // TODO(b/319297222) Modify element such that we render each
                // feature in the feature salience module in a separate line.
                salienceContent : html`
                <span class='salience-placeholder'>
                  Select a datapoint to see ${name} attributions.
                </span>`}
              ${this.state[name].isLoading ? this.renderSpinner() : null}
            </div>
          </div>
        </expansion-panel>
      </div>`;
    // clang-format on
  }

  override renderImpl() {
    // clang-format off
    return html`
      <div class='module-container'>
        ${this.renderHeaderControls()}
        <div class='module-results-area ${SCROLL_SYNC_CSS_CLASS}'>
          ${Object.keys(this.state).map(name => this.renderMethodRow(name))}
        </div>
        ${this.isRenderImage ? null : this.renderFooter()}
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    // TODO(b/204779018): Add appState interpreters to method arguments.

    // Ensure there are salience interpreters for loaded models.
    const appState = app.getService(AppState);
    if (appState.metadata == null) return false;

    return Object.values(modelSpecs).some(modelInfo =>
        modelInfo.interpreters.some(name => {
          const interpreter = appState.metadata.interpreters[name];
          const salienceKeys =
              findSpecKeys(interpreter.metaSpec, SUPPORTED_SALIENCE_TYPES);
          return salienceKeys.length > 0;
        }));
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'salience-map-module': SalienceMapModule;
  }
}
