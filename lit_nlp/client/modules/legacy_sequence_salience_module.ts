/**
 * @fileoverview Visualization for seq2seq salience maps.
 *
 * DEPRECATED; use sequence_salience_module.ts instead.
 */

import '../elements/switch';
import '../elements/numeric_input';
import '../elements/token_chips';

// tslint:disable:no-new-decorators
import {html} from 'lit';
import {customElement} from 'lit/decorators.js';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {TokenWithWeight} from '../elements/token_chips';
import {SignedSalienceCmap, UnsignedSalienceCmap} from '../lib/colors';
import {SequenceSalienceMap} from '../lib/dtypes';
import {canonicalizeGenerationResults, type GeneratedTextResult, GENERATION_TYPES, getAllTargetOptions, TargetOption} from '../lib/generated_text_utils';
import {Salience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {type IndexedInput, ModelInfoMap, type Spec} from '../lib/types';
import {sumArray} from '../lib/utils';

import {styles} from './legacy_sequence_salience_module.css';

interface TokenFocusState {
  idx: number; /* output token index */
  sticky: boolean;
}

// Modes for color scaling.
enum ColorScalingMode {
  NORMALIZE = 'normalize',
  MAX_LOCAL = 'max_local',
  MAX_GLOBAL = 'max_global',
}

const LEGEND_INFO_TITLE_SIGNED =
    'Salience is relative to the model\'s prediction of a class. A positive ' +
    'score (more green) for a token means that token influenced the model to ' +
    'predict that class, whereas a negaitve score (more pink) means the ' +
    'token influenced the model to not predict that class.';

const LEGEND_INFO_TITLE_UNSIGNED =
    'Salience is relative to the model\'s prediction of a class. A larger ' +
    'score (more purple) for a token means that token was more influential ' +
    'on the model\'s prediction of that class.';

/** LIT module for model output. */
@customElement('legacy-sequence-salience-module')
export class LegacySequenceSalienceModule extends LitModule {
  static override title = 'Sequence Salience (legacy)';
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = true;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`
  <legacy-sequence-salience-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </legacy-sequence-salience-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Current data
  @observable private currentData?: IndexedInput;
  @observable private currentPreds?: GeneratedTextResult;
  @observable private salienceTarget?: string;
  @observable
  private currentSalience: {[fieldName: string]: SequenceSalienceMap} = {};
  @observable private selectedSalienceField?: string = undefined;
  @observable private focusState?: TokenFocusState = undefined;

  // Options
  @observable private cmapGamma: number = 2.0;
  @observable
  private cmapScalingMode: ColorScalingMode = ColorScalingMode.NORMALIZE;
  @observable private denseView: boolean = false;

  @computed
  get salienceSpecInfo(): Spec {
    return this.appState.metadata.interpreters['sequence_salience']?.metaSpec ??
        {};
  }

  @computed
  get cmap() {
    if (this.selectedSalienceField != null &&
        (this.salienceSpecInfo[this.selectedSalienceField] as Salience)
            .signed) {
      return new SignedSalienceCmap(/* gamma */ this.cmapGamma);
    } else {
      return new UnsignedSalienceCmap(/* gamma */ this.cmapGamma);
    }
  }

  @computed
  get salienceTargetOptions(): TargetOption[] {
    const dataSpec = this.appState.currentDatasetSpec;
    const outputSpec = this.appState.getModelSpec(this.model).output;
    return getAllTargetOptions(
        dataSpec, outputSpec, this.currentData, this.currentPreds);
  }

  override firstUpdated() {
    if (this.selectedSalienceField === undefined) {
      this.selectedSalienceField = Object.keys(this.salienceSpecInfo)[0];
    }

    const getPrimarySelectedInputData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getPrimarySelectedInputData, async data => {
      this.focusState = undefined; /* clear focus */
      await this.updateToSelection(data);
      this.salienceTarget = this.salienceTargetOptions[0].text;
    });

    this.reactImmediately(() => this.salienceTarget, target => {
      this.updateSalience(this.currentData, target);
    });
  }

  private async updateToSelection(input: IndexedInput|null) {
    if (input == null) {
      this.currentData = undefined;
      this.currentPreds = undefined;
      return;
    }
    // Before waiting for the backend call, update data and clear annotations.
    this.currentData = input;
    this.currentPreds = undefined;

    const promise = this.apiService.getPreds(
        [input], this.model, this.appState.currentDataset, GENERATION_TYPES, [],
        'Getting targets from model prediction');
    const results = await this.loadLatest('generationResults', promise);
    if (results === null) return;

    const outputSpec = this.appState.getModelSpec(this.model).output;
    const preds = canonicalizeGenerationResults(results[0], outputSpec);

    // Update data again, in case selection changed rapidly.
    this.currentData = input;
    this.currentPreds = preds;
  }

  private async updateSalience(input?: IndexedInput, targetText?: string) {
    if (input == null || targetText == null) {
      this.currentSalience = {};
      return;
    }
    // Before waiting for the backend call, clear annotations.
    this.currentSalience = {};

    const salienceTargetConfig = {'target_text': [targetText]};
    const promise = this.apiService.getInterpretations(
        [input], this.model, this.appState.currentDataset, 'sequence_salience',
        salienceTargetConfig, `Computing sequence salience`);
    const results = await this.loadLatest('salience', promise);
    if (results === null) return;

    this.currentSalience = results[0];
  }


  renderField(fieldName: string) {
    const preds = this.currentSalience[fieldName];

    // Output token to show salience for.
    const focusIdx = this.focusState?.idx ?? -1;

    const cmap = this.cmap;
    // length is preds.tokens_in.length + preds.tokens_out.length
    const salience: number[] = focusIdx > -1 ? preds.salience[focusIdx] :
                                               [...preds.salience[0]].fill(0);

    // Apply appropriate scaling.
    // TODO(lit-dev): make this show up in the colormap?
    let denominator = 1.0;
    if (this.cmapScalingMode === ColorScalingMode.NORMALIZE) {
      denominator = sumArray(salience.map(Math.abs));
    } else if (this.cmapScalingMode === ColorScalingMode.MAX_LOCAL) {
      denominator = Math.max(...salience.map(Math.abs));
    } else if (this.cmapScalingMode === ColorScalingMode.MAX_GLOBAL) {
      const perTokenMax =
          preds.salience.map(svals => Math.max(...svals.map(Math.abs)));
      denominator = Math.max(...perTokenMax);
    }
    const scaledSalience =
        denominator > 0 ? salience.map(v => v / denominator) : salience;

    const onMouseoverTarget = (idx: number) => {
      if (this.focusState?.sticky) return;
      this.focusState = {idx, sticky: false};
    };

    const onMouseoutTarget = () => {
      if (this.focusState?.sticky) return;
      this.focusState = undefined;
    };

    const onClickTarget = (idx: number, e: Event) => {
      e.stopPropagation(); /* don't trigger focus clear from background */
      if (this.focusState?.idx === idx && this.focusState?.sticky) {
        // Only clear if match and sticky, since usually this will already
        // be the non-sticky focus due to mouseover.
        this.focusState = undefined; /* clear focus */
      } else {
        this.focusState = {idx, sticky: true};
      }
    };

    const sourceTokensWithWeights: TokenWithWeight[] =
        preds.tokens_in.map((token: string, i: number) => {
          return {
            token,
            weight: scaledSalience[i],
            disableHover: focusIdx === -1
          };
        });

    const targetTokensWithWeights: TokenWithWeight[] =
        preds.tokens_out.map((token: string, i: number) => {
          return {
            token,
            weight: scaledSalience[i + preds.tokens_in.length],
            selected: i === focusIdx,
            pinned: i === focusIdx && (this.focusState?.sticky === true),
            onClick: (e: Event) => {
              onClickTarget(i, e);
            },
            onMouseover: () => {
              onMouseoverTarget(i);
            },
            onMouseout: onMouseoutTarget,
          };
        });

    // clang-format off
    return html`
      <div class='field-group'>
        <table class='field-table'>
          <tr>
            <th><div class='subfield-title'>Source</div></th>
            <td><div>
              <lit-token-chips ?dense=${this.denseView}
                .tokensWithWeights=${sourceTokensWithWeights}
                .cmap=${cmap}>
              </lit-token-chips>
            </div></td>
          </tr>
          <tr>
            <th><div class='subfield-title'>Target</div></th>
            <td><div>
              <lit-token-chips ?dense=${this.denseView}
                .tokensWithWeights=${targetTokensWithWeights}
                .cmap=${cmap}>
              </lit-token-chips>
            </div></td>
          </tr>
        </table>
      </div>
    `;
    // clang-format on
  }

  renderContent() {
    if (!this.currentData) return null;
    if (this.selectedSalienceField === undefined) return null;
    if (!this.currentSalience[this.selectedSalienceField]) return null;

    return this.renderField(this.selectedSalienceField);
  }

  renderHeaderControls() {
    const onChangeTarget = (e: Event) => {
      this.salienceTarget = (e.target as HTMLInputElement).value;
    };

    const onChangeMethod = (e: Event) => {
      this.selectedSalienceField = (e.target as HTMLInputElement).value;
    };

    const onChangeScalingMode = (e: Event) => {
      this.cmapScalingMode =
          (e.target as HTMLInputElement).value as ColorScalingMode;
    };

    // clang-format off
    return html`
      <div class="controls-group controls-group-variable" title="Target string for salience.">
        <label class="dropdown-label">Target:</label>
        <select class="dropdown" @change=${onChangeTarget}>
          ${this.salienceTargetOptions.map(target =>
            html`<option value=${target.text}
                  ?selected=${target.text === this.salienceTarget}>
                   (${target.source}) "${target.text}"
                 </option>`)}
        </select>
      </div>
      <div class="controls-group">
        <div title="Type of salience map to use.">
          <label class="dropdown-label">Method:</label>
          <select class="dropdown" @change=${onChangeMethod}>
            ${Object.keys(this.salienceSpecInfo).map(k =>
              html`<option value=${k} ?selected=${k === this.selectedSalienceField}>
                     ${k}
                   </option>`)}
          </select>
        </div>
        <div title="Method to scale raw values for display.">
          <label class="dropdown-label">Scaling:</label>
          <select class="dropdown" @change=${onChangeScalingMode}>
            ${Object.values(ColorScalingMode).map((mode: ColorScalingMode) =>
              html`<option value=${mode} ?selected=${mode === this.cmapScalingMode}>
                     ${mode}
                   </option>`)}
          </select>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderColorLegend() {
    const cmap = this.cmap;
    const isSigned = (cmap instanceof SignedSalienceCmap);
    const labelName = 'Token Salience';

    const tooltipText =
        isSigned ? LEGEND_INFO_TITLE_SIGNED : LEGEND_INFO_TITLE_UNSIGNED;

    // clang-format off
    return html`
      <color-legend legendType=${LegendType.SEQUENTIAL}
        label=${labelName}
        alignRight
        .scale=${cmap.asScale()}
        numBlocks=${isSigned ? 7 : 5}
        .paletteTooltipText=${tooltipText}>
      </color-legend>`;
    // clang-format on
  }

  renderFooterControls() {
    const onClickToggleDensity = () => {
      this.denseView = !this.denseView;
    };

    const onChangeGamma = (e: Event) => {
      this.cmapGamma = +((e.target as HTMLInputElement).value);
    };

    // clang-format off
    return html`
      <div class="controls-group">
        <lit-switch labelLeft="Dense view"
              ?selected=${this.denseView}
              @change=${onClickToggleDensity}>
        </lit-switch>
      </div>
      <div class="controls-group">
        ${this.renderColorLegend()}
        <label for="gamma-slider">Gamma:</label>
        <lit-numeric-input min="0.25" max="6" step="0.25"
          value="${this.cmapGamma}" @change=${onChangeGamma}>
        </lit-numeric-input>
      </div>`;
    // clang-format on
  }

  override renderImpl() {
    const clearStickyFocus = () => {
      if (this.focusState?.sticky) {
        this.focusState = undefined; /* clear focus */
      }
    };

    // clang-format off
    return html`
      <div class="module-container">
        <div class="module-toolbar">
          ${this.renderHeaderControls()}
        </div>
        <div class="module-results-area"
         @click=${clearStickyFocus}>
          ${this.renderContent()}
        </div>
        <div class="module-footer">
          ${this.renderFooterControls()}
        </div>
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    for (const modelInfo of Object.values(modelSpecs)) {
      if (modelInfo.interpreters.indexOf('sequence_salience') !== -1) {
        return true;
      }
    }
    return false;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'legacy-sequence-salience-module': LegacySequenceSalienceModule;
  }
}
