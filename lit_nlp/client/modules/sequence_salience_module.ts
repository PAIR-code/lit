/**
 * @fileoverview Visualization for seq2seq salience maps.
 */

import '@material/mwc-switch';
import '../elements/slider';

// tslint:disable:no-new-decorators
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {canonicalizeGenerationResults, GeneratedTextResult, GENERATION_TYPES, getAllOutputTexts, getAllReferenceTexts} from '../lib/generated_text_utils';
import {Salience} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {sumArray} from '../lib/utils';
import {SignedSalienceCmap, UnsignedSalienceCmap} from '../services/color_service';

import {styles} from './sequence_salience_module.css';

interface SequenceSalienceMap {
  sourceTokens: string[];
  targetTokens: string[];
  /** [targetTokens.length, sourceTokens.length + targetTokens.length] */
  salience: number[][];
}

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
    "Salience is relative to the model's prediction of a class. A positive " +
    "score (more green) for a token means that token influenced the model to " +
    "predict that class, whereas a negaitve score (more pink) means the " +
    "token influenced the model to not predict that class.";

const LEGEND_INFO_TITLE_UNSIGNED =
    "Salience is relative to the model's prediction of a class. A larger " +
    "score (more purple) for a token means that token was more influential " +
    "on the model's prediction of that class.";

/** LIT module for model output. */
@customElement('sequence-salience-module')
export class SequenceSalienceModule extends LitModule {
  static override title = 'Sequence Salience';
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = true;
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
  <sequence-salience-module model=${model} .shouldReact=${shouldReact}
    selectionServiceIndex=${selectionServiceIndex}>
  </sequence-salience-module>`;

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
  @observable private showValues: boolean = false;

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
  get salienceTargetStrings(): string[] {
    const dataSpec = this.appState.currentDatasetSpec;
    const outputSpec = this.appState.getModelSpec(this.model).output;
    const ret = getAllReferenceTexts(dataSpec, outputSpec, this.currentData);
    if (this.currentData != null && this.currentPreds != null) {
      ret.push(...getAllOutputTexts(outputSpec, this.currentPreds));
    }
    return ret;
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
      this.salienceTarget = this.salienceTargetStrings[0];
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

    const preds = results[0];
    const processedPreds: {[fieldName: string]: SequenceSalienceMap} = {};
    for (const fieldName of Object.keys(preds)) {
      processedPreds[fieldName] = {
        sourceTokens: preds[fieldName]['tokens_in'],
        targetTokens: preds[fieldName]['tokens_out'],
        salience: preds[fieldName]['salience'],
      };
    }
    // Update data again, in case selection changed rapidly.
    this.currentSalience = processedPreds;
  }


  renderField(fieldName: string) {
    const preds = this.currentSalience[fieldName];

    // Output token to show salience for.
    const focusIdx = this.focusState?.idx ?? -1;

    const cmap = this.cmap;
    // length is preds.sourceTokens.length + preds.targetTokens.length
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

    const holderClasses = classMap({
      'token-holder': true,
      'token-holder-dense': this.denseView,
    });

    // Hide values for unimportant tokens, based on current colormap scale.
    // 1 - lightness is roughly how dark the token highlighting will be;
    // 0.05 is an arbitrary threshold that seems to work well.
    function showNumber(val: number) {return 1 - cmap.lightness(val) > 0.05;}
    function displayVal(val: number) {return val.toFixed(3);}
    // TODO(b/204887716, b/173469699): improve layout density?
    // Try to strip leading zeros to save some space? Can do down to 24px width
    // with this, but need to handle negatives.
    // const displayVal = (val: number) => {
    //   const formattedVal = val.toFixed(3);
    //   if (formattedVal[0] === '0') {
    //     return formattedVal.slice(1);
    //   }
    //   return formattedVal;
    // };

    const renderSourceToken = (token: string, i: number) => {
      const val = scaledSalience[i];
      const tokenStyle = styleMap(
          {'color': cmap.textCmap(val), 'background-color': cmap.bgCmap(val)});
      const tokenClasses = classMap({
        'salient-token': true,
        'salient-token-with-number': this.showValues,
      });
      // clang-format off
      return html`
        <div class=${tokenClasses} style=${tokenStyle}
         data-displayval=${displayVal(val)} ?data-shownumber=${showNumber(val)}>
          ${token}
        </div>
      `;
      // clang-format on
    };

    const renderTargetToken = (token: string, i: number) => {
      const val = scaledSalience[i + preds.sourceTokens.length];
      const tokenStyle = styleMap(
          {'color': cmap.textCmap(val), 'background-color': cmap.bgCmap(val)});
      const tokenClasses = classMap({
        'salient-token': true,
        'salient-token-with-number': this.showValues,
        'target-token': true,
        'token-focused': i === focusIdx,
        'token-pinned': i === focusIdx && (this.focusState?.sticky === true),
      });
      // clang-format off
      return html`
        <div class=${tokenClasses} style=${tokenStyle}
         data-displayval=${displayVal(val)} ?data-shownumber=${showNumber(val)}
         @mouseover=${() => { onMouseoverTarget(i); }}
         @mouseout=${onMouseoutTarget}
         @click=${(e: Event) => { onClickTarget(i, e); }}>
          ${token}
        </div>
      `;
      // clang-format on
    };

    // clang-format off
    return html`
      <div class='field-group'>
        <table class='field-table'>
          <tr>
            <th><div class='subfield-title'>Source</div></th>
            <td><div class=${holderClasses}>
                ${preds.sourceTokens.map(renderSourceToken)}
            </div></td>
          </tr>
          <tr>
            <th><div class='subfield-title'>Target</div></th>
             <td><div class=${holderClasses}>
                ${preds.targetTokens.map(renderTargetToken)}
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
          ${this.salienceTargetStrings.map(target =>
            html`<option value=${target} ?selected=${target === this.salienceTarget}>
                   ${target}
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
    function scale(val: number) {return cmap.bgCmap(val);}
    scale.domain = () => cmap.colorScale.domain();
    const labelName = "Token Salience";

    // clang-format off
    return html`<div class="color-legend-container">
      <color-legend legendType=${LegendType.SEQUENTIAL}
        selectedColorName=${labelName}
        ?alignRight=${true}
        .scale=${scale}
        numBlocks=${isSigned ? 7 : 5}>
      </color-legend>
      <mwc-icon class="icon material-icon-outlined"
                title=${isSigned ? LEGEND_INFO_TITLE_SIGNED :
                                   LEGEND_INFO_TITLE_UNSIGNED}>
        info_outline
      </mwc-icon>
    </div>`;
    // clang-format on
  }

  renderFooterControls() {
    const onClickToggleDensity = () => {
      this.denseView = !this.denseView;
    };

    const onClickToggleValues = () => {
      this.showValues = !this.showValues;
    };

    const onChangeGamma = (e: Event) => {
      this.cmapGamma = +((e.target as HTMLInputElement).value);
    };

    // clang-format off
    return html`
      <div class="controls-group">
        <div class='switch-container' @click=${onClickToggleDensity}>
          <div>Dense view</div>
          <mwc-switch ?selected=${this.denseView}></mwc-switch>
        </div>
        <div class='vertical-separator'></div>
        <div class='switch-container' @click=${onClickToggleValues}>
          <div>Show values</div>
          <mwc-switch ?selected=${this.showValues}></mwc-switch>
        </div>
      </div>
      <div class="controls-group">
        ${this.renderColorLegend()}
        <label for="gamma-slider">Gamma:</label>
        <lit-slider min="0.25" max="6" step="0.25" val="${this.cmapGamma}"
                    .onInput=${onChangeGamma}></lit-slider>
        <div class="gamma-value">${this.cmapGamma.toFixed(2)}</div>
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

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
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
    'sequence-salience-module': SequenceSalienceModule;
  }
}
