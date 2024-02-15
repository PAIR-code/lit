/**
 * @fileoverview Custom viz module for causal LM salience.
 */

import '@material/mwc-icon';
import '../elements/color_legend';
import '../elements/numeric_input';
import '../elements/fused_button_bar';

import {css, html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators.js';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {NumericInput as LitNumericInput} from '../elements/numeric_input';
import {TokenChips, TokenWithWeight} from '../elements/token_chips';
import {SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap,} from '../lib/colors';
import {GENERATION_TYPES, getAllTargetOptions, TargetOption, TargetSource} from '../lib/generated_text_utils';
import {LitType, LitTypeTypesList, Tokens, TokenScores} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {cleanSpmText, groupTokensByRegexPrefix} from '../lib/token_utils';
import {type IndexedInput, type Preds, SCROLL_SYNC_CSS_CLASS, type Spec} from '../lib/types';
import {cumSumArray, filterToKeys, findSpecKeys, groupAlike, makeModifiedInput, sumArray} from '../lib/utils';

import {styles} from './lm_salience_module.css';

/**
 * Max of absolute value
 */
export function maxAbs(vals: number[]): number {
  return Math.max(...vals.map(Math.abs));
}

enum SegmentationMode {
  TOKENS = 'Tokens',
  WORDS = 'Words',
  SENTENCES = 'Sentences',
  LINES = 'Lines',
  // TODO(b/324961811): add phrase or clause chunking?
  // TODO(b/324961803): add custom regex?
}

const LEGEND_INFO_TITLE_SIGNED =
    'Salience is relative to the model\'s prediction of a token. A positive ' +
    'score (more green) for a token means that token influenced the model to ' +
    'predict the selected target, whereas a negaitve score (more pink) means ' +
    'the token influenced the model to not predict the selected target.';

const LEGEND_INFO_TITLE_UNSIGNED =
    'Salience is relative to the model\'s prediction of a token. A larger ' +
    'score (more purple) for a token means that token was more influential ' +
    'on the model\'s prediction of the selected target.';

/**
 * A convenience implementation of LitModule for single model, single example
 * use. Implements some standard boilerplate to fetch model predictions.
 *
 * Subclass should still register this with @customElement, and add to the
 * HTMLElementTagNameMap, we well as implement:
 * - static template = ...
 * - override renderImpl() {...}
 *
 * And optionally:
 * - static styles() {...}
 * - static override shouldDisplayModule() {...}
 *
 * If subclass implements firstUpdated(), be sure to call super.firstUpdated()
 * to register the reaction to the primary selection.
 */
export class SingleExampleSingleModelModule extends LitModule {
  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = true;

  // Override this to request only specific types.
  protected predsTypes: LitTypeTypesList = [LitType];

  @observable protected currentData?: IndexedInput;
  @observable protected currentPreds?: Preds;

  // Override this for any post-processing.
  protected postprocessPreds(input: IndexedInput, preds: Preds): Preds {
    return preds;
  }

  protected resetState() {
    this.currentData = undefined;
    this.currentPreds = undefined;
  }

  protected async updateToSelection() {
    this.resetState();

    const input = this.selectionService.primarySelectedInputData;
    if (input == null) return;

    // Before waiting for the backend call, update data.
    // currentPreds should already be cleared by the resetState() call above.
    this.currentData = input;

    const promise = this.apiService.getPreds(
        [input],
        this.model,
        this.appState.currentDataset,
        this.predsTypes,
        [],
        `Getting predictions from ${this.model}`,
    );
    const results = await this.loadLatest('modelPreds', promise);
    if (results === null) return;

    const preds = this.postprocessPreds(input, results[0]);

    // Update data again, in case selection changed rapidly.
    this.currentData = input;
    this.currentPreds = preds;
  }

  override firstUpdated() {
    this.reactImmediately(
        () =>
            [this.selectionService.primarySelectedInputData, this.model,
             this.appState.currentDataset],
        () => {
          this.updateToSelection();
        },
    );
  }
}

/**
 * Custom styled version of <lit-token-chips> for rendering LM salience tokens.
 */
@customElement('lm-salience-chips')
class LMSalienceChips extends TokenChips {
  static override get styles() {
    return [
      ...TokenChips.styles,
      css`
        .salient-token {
          padding: 1px 3px; /* wider horizontally */
          margin: 2px;
          min-width: 4px;  /* easier to see whitespace tokens */
        }
        .tokens-holder:not(.tokens-holder-dense) .salient-token:not(.selected) {
          --token-outline-color: var(--lit-neutral-300); /* outline in non-dense mode */
        }
        .tokens-holder-display-block .salient-token {
          padding: 3px 0;
          margin: 0;
          margin-right: 4px;
        }
        .salient-token.selected {
          --token-outline-color: var(--lit-mage-700);
          box-shadow: 0px 0px 3px var(--token-outline-color);
        }
        .tokens-holder-dense .salient-token {
          margin: 2px 0px;  /* vertical spacing only */
          min-width: 6px;  /* not too small. Check if this causes issues inside words. */
        }
        .tokens-holder-dense .salient-token.selected {
          outline: 2px solid var(--token-outline-color);
          border: 0;
          box-shadow: unset;
          /* TODO see if we can get away from z-index here */
          z-index: 1;
        }
      `,
    ];
  }
}

interface SalienceResults {
  [method: string]: number[];
}

// Sentinel value because mobx doesn't react directly to a promise completing.
const REQUEST_PENDING: unique symbol = Symbol('REQUEST_PENDING');

/** LIT module for model output. */
@customElement('lm-salience-module')
export class LMSalienceModule extends SingleExampleSingleModelModule {
  static override title = 'LM Salience';
  static override numCols = 6;  // 60% of screen width if DataTable on left
  static override duplicateAsRow = true;
  // prettier-ignore
  static override template = (
      model: string,
      selectionServiceIndex: number,
      shouldReact: number,
      ) => html`<lm-salience-module model=${model} .shouldReact=${shouldReact}
                  selectionServiceIndex=${selectionServiceIndex}>
                </lm-salience-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  // For generation model. For salience, see updateSalience() below.
  override predsTypes = GENERATION_TYPES;

  @observable
  private segmentationMode: SegmentationMode = SegmentationMode.WORDS;
  // TODO(b/324959547): get default from spec
  @observable private selectedSalienceMethod? = 'grad_l2';
  @observable private cmapGamma = 1.0;
  @observable private denseView = true;
  @observable private showSelfSalience = false;

  @observable.ref private currentTokens: string[] = [];
  @observable.ref private salienceTargetOptions: TargetOption[] = [];
  @observable private salienceTargetString = '';
  @observable.ref private targetSegmentSpan?: [number, number] = undefined;


  /**
   * Cache for salience results for different target spans.
   * Because computing salience can be slow and we don't want to lock the
   * frontend, we use this cache as an intermediary between the API calls
   * (updateSalience) and the rendering logic. API calls are asynchronous with
   * updates and populate this cache with their results; the rendering logic
   * then observes this cache and renders only the result with the current
   * selected span.
   *
   * Each cache entry can have three states:
   * - undefined: we haven't seen it yet, so updateSalience will issue a backend
   * call.
   * - REQUEST_PENDING: sentinel value, set while a backend call is in progress.
   * - Otherwise, will contain a SalienceResults object with results for that
   * key.
   */
  @observable
  private salienceResultCache:
      {[targetKey: string]: SalienceResults|(typeof REQUEST_PENDING)} = {};

  @computed
  get salienceModelName(): string {
    return `_${this.model}_salience`;
  }

  @computed
  get tokenizerModelName(): string {
    // TODO: fall back to salience model if not available?
    return `_${this.model}_tokenizer`;
  }

  private resetTargetSpan() {
    this.targetSegmentSpan = undefined;
  }

  override resetState() {
    // Generation & target string selection
    super.resetState();  // currentData and currentPreds
    this.salienceTargetOptions = [];
    this.salienceTargetString = '';
    // Tokens and selected target span
    this.currentTokens = [];
    this.resetTargetSpan();
    // Salience results
    this.salienceResultCache = {};
  }

  // Get generations; populate this.currentPreds
  protected override async updateToSelection() {
    await super.updateToSelection();
    this.resetTargetSpan();

    const dataSpec = this.appState.currentDatasetSpec;
    const outputSpec = this.appState.getModelSpec(this.model).output;
    this.salienceTargetOptions = getAllTargetOptions(
        dataSpec,
        outputSpec,
        this.currentData,
        this.currentPreds,
    );
    this.salienceTargetString = this.salienceTargetOptions[0]?.text ?? '';
  }

  // Modified input with selected target sequence. Use this for tokens and
  // salience.
  @computed
  get modifiedData(): IndexedInput|null {
    if (this.currentData == null) return null;
    return makeModifiedInput(
        this.currentData, {'target': this.salienceTargetString});
  }

  @computed
  get currentTokenGroups(): string[][] {
    if (this.segmentationMode === SegmentationMode.TOKENS) {
      return this.currentTokens.map(t => [t]);
    } else if (this.segmentationMode === SegmentationMode.WORDS) {
      // Word start is either:
      // - whitespace or magic underscore
      // - any non-\n following \n
      // The latter is needed to avoid forming weird segments like '\n\nfoo';
      // by using the lookbehind, this will end up as ['\n\n', 'foo']
      return groupTokensByRegexPrefix(
          this.currentTokens, /([▁\s]+)|(?<=\n)[^\n]/g);
    } else if (this.segmentationMode === SegmentationMode.SENTENCES) {
      // Sentence start is one of:
      // - a run of consecutive \n as its own segment
      // - any non-\n following \n
      // - whitespace or magic underscore following punctuation [.?!]
      return groupTokensByRegexPrefix(
          this.currentTokens, /(\n+)|((?<=\n)[^\n])|((?<=[.?!])([▁\s]+))/g);
    } else if (this.segmentationMode === SegmentationMode.LINES) {
      // Line start is either:
      // - a run of consecutive \n as its own segment
      // - any non-\n following \n
      return groupTokensByRegexPrefix(this.currentTokens, /(\n+)|([^\n]+)/g);
    } else {
      throw new Error(
          `Unsupported segmentation mode ${this.segmentationMode}.`);
    }
  }

  /**
   * Segment offsets, as token indices.
   * Segment i corresponds to tokens offsets[i]:offsets[i+1]
   */
  @computed
  get currentSegmentOffsets(): number[] {
    return [0, ...cumSumArray(this.currentTokenGroups.map(g => g.length))];
  }

  @computed
  get targetTokenSpan(): number[]|undefined {
    if (this.targetSegmentSpan === undefined) return undefined;
    const [segmentStart, segmentEnd] = this.targetSegmentSpan;
    const offsets = this.currentSegmentOffsets;
    return [offsets[segmentStart], offsets[segmentEnd]];
  }

  @computed
  get currentSegmentTexts(): string[] {
    const segments = this.currentTokenGroups.map(tokens => tokens.join(''));
    // Tokens in non-dense view should show exact tokenization, including magic
    // underscores.
    if (this.segmentationMode === SegmentationMode.TOKENS && !this.denseView) {
      return segments;
    }
    // Otherwise, clean up underscores.
    return segments.map(cleanSpmText);
  }

  @computed
  get salienceSpecInfo(): Spec {
    const outputSpec =
        this.appState.getModelSpec(this.salienceModelName).output;
    const salienceKeys = findSpecKeys(outputSpec, TokenScores);
    return filterToKeys(outputSpec, salienceKeys);
  }

  /**
   * Salience for active model, for all tokens.
   */
  @computed
  get activeTokenSalience(): number[]|undefined {
    if (this.targetTokenSpan === undefined) return undefined;

    const cachedResult =
        this.salienceResultCache[this.spanToKey(this.targetTokenSpan)];
    if (cachedResult === undefined || cachedResult === REQUEST_PENDING) {
      return undefined;
    }

    if (this.selectedSalienceMethod === undefined) {
      return undefined;
    }

    return cachedResult[this.selectedSalienceMethod];
  }

  /**
   * Salience for active mode, for current segments.
   */
  @computed
  get activeSalience(): number[]|undefined {
    if (this.activeTokenSalience === undefined) return undefined;
    const groupedSalience =
        groupAlike(this.activeTokenSalience, this.currentTokenGroups);
    return groupedSalience.map(sumArray);
  }

  @computed
  get cmapRange(): number {
    if (this.activeSalience === undefined) return 1;
    // If nothing focused, use the max over all (absolute) scores.
    return Math.max(1e-3, maxAbs(this.activeSalience));
  }

  @computed
  get signedSalienceCmap() {
    return new SignedSalienceCmap(this.cmapGamma, [
      -1 * this.cmapRange,
      this.cmapRange,
    ]);
  }

  @computed
  get unsignedSalienceCmap() {
    return new UnsignedSalienceCmap(this.cmapGamma, [0, this.cmapRange]);
  }

  @computed
  get cmap(): SalienceCmap {
    // TODO(b/324959547): get signed/unsigned info from spec.
    // May need to add a signed= bit to the TokenScores type,
    // or use the TokenSalience type.
    return this.selectedSalienceMethod === 'grad_dot_input' ?
        this.signedSalienceCmap :
        this.unsignedSalienceCmap;
  }

  spanToKey(span: number[]) {
    return `${span[0]}:${span[1]}`;
  }

  async updateTokens() {
    this.currentTokens = [];

    const input = this.modifiedData;
    if (input == null) {
      return;
    }

    const promise = this.apiService.getPreds(
        [input],
        this.tokenizerModelName,
        this.appState.currentDataset,
        [Tokens],
        [],
        `Fetching tokens for model ${this.model}`,
    );
    const results = await this.loadLatest('updateTokens', promise);
    if (results === null) {
      console.warn('No tokens returned or stale request for example', input);
      return;
    }

    // TODO(b/324959547): get field name from spec, rather than hardcoding
    // 'tokens'.
    this.currentTokens = results[0]['tokens'];
  }

  async updateSalience(targetTokenSpan: number[]|undefined) {
    if (this.modifiedData == null) return;
    if (targetTokenSpan === undefined) return;

    const spanKey = this.spanToKey(targetTokenSpan);
    const cachedResult = this.salienceResultCache[spanKey];
    if (cachedResult !== undefined) {
      if (cachedResult === REQUEST_PENDING) {
        // Another call is waiting and we can let that update the results.
        console.log('Duplicate request for target span ', spanKey);
      } else {
        // Actual results.
        console.log('Found cached return for target span ', spanKey);
      }
      // No need to proceed with backend call in either case.
      return;
    }

    this.salienceResultCache[spanKey] = REQUEST_PENDING;

    const [start, end] = targetTokenSpan;
    const targetMask = this.currentTokens.map(
        (t: string, i) => (i >= start && i < end) ? 1 : 0);

    // TODO(b/324959547): don't hard-code 'target_mask', get field name from
    // spec. We may want to create a custom TargetMask type for this.
    const maskedData = makeModifiedInput(
        this.modifiedData, {'target_mask': targetMask}, 'salience');

    const promise = this.apiService.getPreds(
        [maskedData],
        this.salienceModelName,
        this.appState.currentDataset,
        [TokenScores],
        [],
        `Getting salience scores for ${this.printTargetForHuman(start, end)}`,
    );
    const results = await promise;
    if (results === null) {
      console.warn('Empty results from request', maskedData, spanKey);
      delete this.salienceResultCache[spanKey];
      return;
    }

    this.salienceResultCache[spanKey] = results[0];
  }

  override firstUpdated() {
    super.firstUpdated();

    // If selected example OR selected target string change.
    // NOTE: you may see a console warning: "Element lm-salience-module
    // scheduled an update (generally because a property was set) after an
    // update completed, causing a new update to be scheduled."
    // This is okay here: this.modifiedData will be updated after
    // updateToSelection() runs, which will trigger this to update tokens.
    this.reactImmediately(
        () => [this.modifiedData, this.model, this.appState.currentDataset],
        () => {
          this.resetTargetSpan();
          this.updateTokens();
        });

    // This can react only to targetTokenSpan, because changes to
    // this.model or this.appState.currentDataset will trigger the target span
    // to be reset.
    this.reactImmediately(() => this.targetTokenSpan, (targetTokenSpan) => {
      this.updateSalience(targetTokenSpan);
    });
  }

  renderGranularitySelector() {
    const onClickToggleDensity = () => {
      this.denseView = !this.denseView;
    };

    const segmentationOptions = Object.values(SegmentationMode).map((val) => {
      return {
        text: val,
        selected: this.segmentationMode === val,
        onClick: () => {
          if (this.segmentationMode !== val) {
            this.targetSegmentSpan = undefined;
          }
          this.segmentationMode = val as SegmentationMode;
        },
      };
    });

    // prettier-ignore
    return html`
      <div class="controls-group" style="gap: 8px;">
        <label class="dropdown-label" for="granularity-selector">Granularity:</label>
        <lit-fused-button-bar id="granularity-selector"
            .options=${segmentationOptions}
            ?disabled=${this.currentTokens.length === 0}>
        </lit-fused-button-bar>
        <lit-switch
          ?selected=${!this.denseView}
          @change=${onClickToggleDensity}>
          <mwc-icon class='icon-button large-icon' title='Flowing text' slot='labelLeft'>
            notes
          </mwc-icon>
          <mwc-icon class='icon-button large-icon' title='Segments' slot='labelRight'>
            grid_view
          </mwc-icon>
        </lit-switch>
      </div>
    `;
  }

  renderSelfScoreSelector() {
    const onClickToggleSelfSalience = () => {
      this.showSelfSalience = !this.showSelfSalience;
    };
    // prettier-ignore
    return html`
      <lit-switch labelLeft="Show self scores"
        ?selected=${this.showSelfSalience}
        @change=${onClickToggleSelfSalience}>
      </lit-switch>
    `;
  }

  renderMethodSelector() {
    const methodOptions = Object.keys(this.salienceSpecInfo).map((key) => {
      return {
        text: key,
        selected: this.selectedSalienceMethod === key,
        onClick: () => {
          if (this.selectedSalienceMethod !== key) {
            this.selectedSalienceMethod = key;
          }
        },
      };
    });

    // prettier-ignore
    return html`
      <div class="controls-group" style="gap: 8px;">
        <label class="dropdown-label" for="method-selector">Method:</label>
        <lit-fused-button-bar .options=${methodOptions} id="method-selector">
        </lit-fused-button-bar>
        ${this.renderSelfScoreSelector()}
      </div>
    `;
  }

  targetSpanText(start: number, end: number): string {
    const tokens = this.currentTokens.slice(start, end);
    // Render text in a way that resembles the way the token chips read
    // at the current display density. Text should match currentSegmentTexts,
    // except:
    //  - Tokens are joined with spaces in non-dense Tokens mode
    //  - Whitespace is trimmed in all other modes
    if (this.segmentationMode === SegmentationMode.TOKENS && !this.denseView) {
      return tokens.join(' ');
    }
    return cleanSpmText(tokens.join('')).trim();
  }

  printTargetForHuman(start: number, end: number): string {
    if (end === start + 1) {
      return `[${start}] "${this.targetSpanText(start, end)}"`;
    } else {
      return `[${start}:${end}] "${this.targetSpanText(start, end)}"`;
    }
  }

  renderSalienceTargetStringSelector() {
    const onChangeTarget = (e: Event) => {
      this.salienceTargetString = (e.target as HTMLInputElement).value;
    };

    const options = this.salienceTargetOptions.map(target => {
      // TODO(b/324959547): get field names 'target' and 'response' from spec
      // via generated_text_utils.ts, rather than hard-coding.
      // This information is available on the frontend, but we need to thread
      // it through a few layers of code in generated_text_utils.ts
      const sourceName =
          target.source === TargetSource.REFERENCE ? 'target' : 'response';
      return html`<option value=${target.text}
            ?selected=${target.text === this.salienceTargetString}>
             (${sourceName}) "${target.text}"
           </option>`;
    });

    // prettier-ignore
    return html`
      <div class="controls-group controls-group-variable"
        title="Target string for salience.">
        <label class="dropdown-label">Target:</label>
        <select class="dropdown" @change=${onChangeTarget}>
          ${options}
        </select>
      </div>`;
  }

  renderTargetIndicator() {
    const printSelectedTargets = () => {
      if (this.targetTokenSpan === undefined) {
        const segmentType = this.segmentationMode === SegmentationMode.TOKENS ?
            'token(s)' :
            'segment(s)';
        // prettier-ignore
        return html`<span class="gray-text">
          Click ${segmentType} above to select a target to explain.
        </span>`;
      }
      const [start, end] = this.targetTokenSpan;
      return `Explaining ${this.printTargetForHuman(start, end)}`;
    };

    // prettier-ignore
    return html`
      <div class="controls-group controls-group-variable"
        title="Selected target span.">
        <div class="target-info-line">
          ${printSelectedTargets()}
        </div>
      </div>
    `;
  }

  /**
   * Set selection (this.targetSegmentSpan) based on current selection and the
   * index of the clicked segment (i).
   */
  private setSegmentTarget(i: number, shiftSelect = false) {
    if (this.targetSegmentSpan === undefined) {
      // If nothing selected, select token i
      this.targetSegmentSpan = [i, i + 1];
      return;
    }
    const [start, end] = this.targetSegmentSpan;
    if (shiftSelect) {
      // Shift: expand target span to this token.
      if (i < start) {
        this.targetSegmentSpan = [i, end];
      } else if (i >= end) {
        this.targetSegmentSpan = [start, i + 1];
      }
      // Otherwise, i is within selection so do nothing.
    } else {
      // Default: only extend by one, otherwise reset.
      if (i === start - 1) {
        // Extend by one token earlier.
        this.targetSegmentSpan = [i, end];
      } else if (i === end) {
        // Extend by one token later.
        this.targetSegmentSpan = [start, i + 1];
      } else if (i === start) {
        // Deselect start token.
        this.targetSegmentSpan = start + 1 < end ? [start + 1, end] : undefined;
      } else if (i === end - 1) {
        // Deselect end token.
        this.targetSegmentSpan = start < end - 1 ? [start, end - 1] : undefined;
      } else {
        // // Interior or discontiguous: select only token i.
        this.targetSegmentSpan = [i, i + 1];
      }
    }
  }

  private inTargetSpan(i: number) {
    if (this.targetSegmentSpan === undefined) return false;
    return i >= this.targetSegmentSpan[0] && i < this.targetSegmentSpan[1];
  }

  renderContent() {
    if (this.currentSegmentTexts.length === 0) return null;

    const segments: string[] = this.currentSegmentTexts;
    const segmentsWithWeights: TokenWithWeight[] = [];
    for (let i = 0; i < segments.length; i++) {
      const selected = this.inTargetSpan(i);
      let weight = this.activeSalience?.[i] ?? 0;
      if (selected && !this.showSelfSalience) {
        weight = 0;
      }
      segmentsWithWeights.push({
        token: segments[i],
        weight,
        selected,
        onClick: (e: MouseEvent) => {
          this.setSegmentTarget(i, e.shiftKey);
          if (e.shiftKey) {
            // Holding shift will also select the token text, which can be
            // distracting. Use this to clear it.
            document.getSelection()?.removeAllRanges();
          }
          e.stopPropagation();
        }
      });
    }

    // TODO: revert to 4px for non-dense view if we can figure out the
    // display mode for token chips? Needs more padding for block mode,
    // but also indentation and newlines are wonky.
    // prettier-ignore
    return html`
      <div class=${this.denseView ? 'chip-container-dense' : 'chip-container'}>
        <lm-salience-chips .tokensWithWeights=${segmentsWithWeights} 
          ?dense=${this.denseView} ?preSpace=${this.denseView}
          .cmap=${this.cmap} breakNewlines displayBlock>
        </lm-salience-chips>
      </div>
    `;
  }

  renderColorLegend() {
    const cmap = this.cmap;
    const isSigned = cmap instanceof SignedSalienceCmap;
    const labelName = 'Salience';

    const tooltipText =
        isSigned ? LEGEND_INFO_TITLE_SIGNED : LEGEND_INFO_TITLE_UNSIGNED;

    // prettier-ignore
    return html`
      <color-legend legendType=${LegendType.SEQUENTIAL}
        label=${labelName}
        .scale=${cmap.asScale()}
        numBlocks=${isSigned ? 7 : 5}
        tooltipPosition="above"
        paletteTooltipText=${tooltipText}>
      </color-legend>`;
  }

  renderColorControls() {
    const onChangeGamma = (e: Event) => {
      // Note: HTMLInputElement.valueAsNumber does not work properly for
      // <lit-numeric-input>
      this.cmapGamma = Number((e.target as LitNumericInput).value);
    };

    const resetGamma = () => {
      this.cmapGamma = 1.0;
    };

    // prettier-ignore
    return html`
      <div class="controls-group">
        ${this.renderColorLegend()}
        <label for="gamma-slider">Colormap intensity:</label>
        <lit-numeric-input min="0" max="6" step="0.25" id='gamma-slider'
          value="${this.cmapGamma}" @change=${onChangeGamma}>
        </lit-numeric-input>
        <mwc-icon class='icon-button value-reset-icon' title='Reset gamma'
          @click=${resetGamma}>
          restart_alt
        </mwc-icon>
      </div>`;
  }

  override renderImpl() {
    const clearTargets = () => {
      this.resetTargetSpan();
    };

    // prettier-ignore
    return html`
      <div class="module-container">
        <div class="module-toolbar">
          ${this.renderSalienceTargetStringSelector()}
        </div>
        <div class="module-toolbar">
          ${this.renderGranularitySelector()}
          ${this.renderMethodSelector()}
        </div>
        <div class="module-results-area ${
        SCROLL_SYNC_CSS_CLASS} flex-column" @click=${clearTargets}>
          ${this.renderContent()}
        </div>
        <div class="module-footer module-footer-wrappable">
          ${this.renderTargetIndicator()}
          ${this.renderColorControls()}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lm-salience-chips': LMSalienceChips;
    'lm-salience-module': LMSalienceModule;
  }
}