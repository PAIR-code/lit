/**
 * @fileoverview Custom viz module for causal LM salience.
 */

import '@material/mwc-icon';
import '../elements/color_legend';
import '../elements/interstitial';
import '../elements/numeric_input';
import '../elements/fused_button_bar';

import {css, html} from 'lit';
// tslint:disable:no-new-decorators
import {customElement, property} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {computed, observable} from 'mobx';

import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {NumericInput as LitNumericInput} from '../elements/numeric_input';
import {TextChips, TokenChips, TokenWithWeight} from '../elements/token_chips';
import {CONTINUOUS_SIGNED_LAB, CONTINUOUS_UNSIGNED_LAB, SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap} from '../lib/colors';
import {GENERATION_TYPES, getAllTargetOptions, TargetOption, TargetSource} from '../lib/generated_text_utils';
import {LitType, LitTypeTypesList, Tokens, TokenScores} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {cleanSpmText, groupTokensByRegexPrefix, groupTokensByRegexSeparator} from '../lib/token_utils';
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
  PARAGRAPHS = '¶',
  // TODO(b/324961811): add phrase or clause chunking?
  CUSTOM = '⚙',
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
class LMSalienceChips extends TextChips {
  @property({type: Boolean}) underline = false;

  override holderClass() {
    return Object.assign(
        {}, super.holderClass(), {'underline': this.underline});
  }

  static override get styles() {
    return [
      ...TokenChips.styles,
      css`
        .text-chips.underline .salient-token {
          --underline-height: 6px;
          background-color: transparent;
          color: black;
        }

        .text-chips.dense.underline .salient-token {
          padding-bottom: var(--underline-height);
        }

        .text-chips.underline .salient-token.selected {
          outline: 1px solid var(--token-outline-color);
          --underline-height: 5px; /* subtract 1px for outline */
        }

        /* In dense mode we style the text span */
        .text-chips.dense.underline .salient-token span {
          /* have to use border- because there is no outline-bottom */
          border-bottom: var(--underline-height) solid var(--token-bg-color);
          border-radius: 2px;
          padding-bottom: 0; /* used by border instead */
        }

        .text-chips.dense.underline .salient-token.selected span {
          /* use mage-500 for underline block as mage-700 is too dark */
          border-bottom: var(--underline-height) solid var(--lit-mage-500);
        }

        /* In non-dense mode we style the containing div */
        .text-chips:not(.dense).underline .salient-token {
          /* have to use border- because there is no outline-bottom */
          border-bottom: var(--underline-height) solid var(--token-bg-color);
          border-radius: 2px;
          padding-bottom: 0; /* used by border instead */
        }

        .text-chips:not(.dense).underline .salient-token.selected {
          /* use mage-500 for underline block as mage-700 is too dark */
          border-bottom: var(--underline-height) solid var(--lit-mage-500);
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

const CMAP_DEFAULT_RANGE = 0.4;

const DEFAULT_CUSTOM_SEGMENTATION_REGEX = '\\n+';

/** LIT module for model output. */
@customElement('lm-salience-module')
export class LMSalienceModule extends SingleExampleSingleModelModule {
  static override title = 'Sequence Salience';
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

  // For generation model. For salience, see fetchSalience() below.
  override predsTypes = GENERATION_TYPES;

  @observable
  private segmentationMode: SegmentationMode = SegmentationMode.WORDS;
  @observable
  private customSegmentationRegexString: string =
      DEFAULT_CUSTOM_SEGMENTATION_REGEX;
  // TODO(b/324959547): get default from spec
  @observable private selectedSalienceMethod? = 'grad_l2';
  // Output range for colormap.
  // cmapDomain is the input range, and is auto-computed from scores below.
  @observable private cmapRange = CMAP_DEFAULT_RANGE;
  @observable private denseView = true;
  @observable private vDense = false; /* vertical spacing */
  @observable private underline = false; /* highlight vs. underline */
  @observable private showSelfSalience = false;

  @observable.ref private currentTokens: string[] = [];
  @observable.ref private salienceTargetOptions: TargetOption[] = [];
  @observable private salienceTargetOption?: number;  // index into above
  @observable.ref private targetSegmentSpan?: [number, number] = undefined;


  /**
   * Cache for salience results for different target spans.
   * Because computing salience can be slow and we don't want to lock the
   * frontend, we use this cache as an intermediary between the API calls
   * (fetchSalience) and the rendering logic. API calls are asynchronous with
   * updates and populate this cache with their results; the rendering logic
   * then observes this cache and renders only the result with the current
   * selected span.
   *
   * Each cache entry can have three states:
   * - undefined: we haven't seen it yet, so fetchSalience will issue a backend
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

  /* be sure to run this when the target string changes */
  private clearResultCache() {
    this.salienceResultCache = {};
  }

  private resetTargetSpan() {
    this.targetSegmentSpan = undefined;
  }

  override resetState() {
    // Generation & target string selection
    super.resetState();  // currentData and currentPreds
    this.salienceTargetOptions = [];
    this.salienceTargetOption = undefined;
    // Tokens and selected target span
    this.currentTokens = [];
    this.resetTargetSpan();
    // Salience results
    this.clearResultCache();
  }

  // Modified input with selected target sequence. Use this for tokens and
  // salience.
  @computed
  get modifiedData(): IndexedInput|null {
    if (this.currentData == null) return null;
    if (this.salienceTargetOption === undefined) return null;
    const targetString =
        this.salienceTargetOptions[this.salienceTargetOption].text;
    return makeModifiedInput(this.currentData, {'target': targetString});
  }

  @computed
  get customSegmentationRegex(): RegExp|undefined {
    try {
      return RegExp(this.customSegmentationRegexString, 'g');
    } catch (error) {
      console.warn(
          'Invalid segmentation regex: ', this.customSegmentationRegexString);
      return undefined;
    }
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
      // Line separator is one or more newlines.
      return groupTokensByRegexSeparator(this.currentTokens, /\n+/g);
    } else if (this.segmentationMode === SegmentationMode.PARAGRAPHS) {
      // Paragraph separator is two or more newlines.
      return groupTokensByRegexSeparator(this.currentTokens, /\n\n+/g);
    } else if (this.segmentationMode === SegmentationMode.CUSTOM) {
      if (this.customSegmentationRegex === undefined) {
        // Just return tokens.
        return this.currentTokens.map(t => [t]);
      } else {
        return groupTokensByRegexPrefix(
            this.currentTokens, this.customSegmentationRegex);
      }
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
  get cmapDomain(): number {
    if (this.activeSalience === undefined) return 1;
    // If nothing focused, use the max over all (absolute) scores.
    return Math.max(1e-3, maxAbs(this.activeSalience));
  }

  @computed
  get cmapGamma(): number {
    // Pin gamma as a function of the range, so we only need a single slider.
    return this.cmapRange * (1.0 / CMAP_DEFAULT_RANGE);
  }

  @computed
  get signedSalienceCmap() {
    return new SignedSalienceCmap(
        this.cmapGamma, [-1 * this.cmapDomain, this.cmapDomain],
        CONTINUOUS_SIGNED_LAB, [0, this.cmapRange]);
  }

  @computed
  get unsignedSalienceCmap() {
    return new UnsignedSalienceCmap(
        this.cmapGamma, [0, this.cmapDomain], CONTINUOUS_UNSIGNED_LAB,
        [0, this.cmapRange]);
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

  async fetchSalience(targetTokenSpan: number[]|undefined) {
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

    // Update target options based on current data and preds.
    // TODO: could this just be @computed?
    // If we maintain explicit state, we can support custom target strings.
    this.reactImmediately(() => [this.currentData, this.currentPreds], () => {
      const dataSpec = this.appState.currentDatasetSpec;
      const outputSpec = this.appState.getModelSpec(this.model).output;
      this.salienceTargetOptions = getAllTargetOptions(
          dataSpec,
          outputSpec,
          this.currentData,
          this.currentPreds,
      );
    });

    // If selected example OR selected target string change.
    // NOTE: you may see a console warning: "Element lm-salience-module
    // scheduled an update (generally because a property was set) after an
    // update completed, causing a new update to be scheduled."
    // This is okay here: this.modifiedData will be updated after
    // updateToSelection() runs, which will trigger this to update tokens.
    this.reactImmediately(
        () => [this.modifiedData, this.model, this.appState.currentDataset],
        () => {
          this.clearResultCache();
          this.resetTargetSpan();
          this.updateTokens();
        });

    // This can react only to targetTokenSpan, because changes to
    // this.model or this.appState.currentDataset will trigger the target span
    // to be reset.
    this.reactImmediately(() => this.targetTokenSpan, (targetTokenSpan) => {
      this.fetchSalience(targetTokenSpan);
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
        tooltipText:
            (val === SegmentationMode.PARAGRAPHS ? 'Paragraphs' :
                 val === SegmentationMode.CUSTOM ? 'Custom Regex' :
                                                   ''),
        onClick: () => {
          if (this.segmentationMode !== val) {
            this.resetTargetSpan();
          }
          this.segmentationMode = val as SegmentationMode;
        },
      };
    });

    const onClickToggleVDense = () => {
      this.vDense = !this.vDense;
    };

    const onClickToggleUnderline = () => {
      this.underline = !this.underline;
    };

    const updateSegmentationRegex = (e: Event) => {
      const {value} = e.target as HTMLInputElement;
      this.customSegmentationRegexString = value;
      this.resetTargetSpan();
    };

    const regexEntryClasses = classMap({
      'regex-input': true,
      // Note: customSegmentationRegex is the RegExp object, it will be
      // undefined if the customSegmentationRegexString does not define a valid
      // regular expression.
      'error-input': this.customSegmentationRegex === undefined
    });

    const resetSegmentationRegex = () => {
      this.customSegmentationRegexString = DEFAULT_CUSTOM_SEGMENTATION_REGEX;
    };

    // prettier-ignore
    const customRegexEntry = html`
      <div class='regex-input-container'>
        <input type='text' class=${regexEntryClasses} slot='tooltip-anchor'
          title="Enter a regex to match segment prefix."
          @input=${updateSegmentationRegex}
          .value=${this.customSegmentationRegexString}>
        <mwc-icon class='icon-button value-reset-icon' title='Reset regex'
          @click=${resetSegmentationRegex}>
          restart_alt
        </mwc-icon>
      </div>
    `;

    // prettier-ignore
    return html`
      <div class="controls-group" style="gap: 8px;">
        <label class="dropdown-label" id="granularity-label"
         for="granularity-selector">Granularity:</label>
        <lit-fused-button-bar id="granularity-selector"
            .options=${segmentationOptions}
            ?disabled=${this.currentTokens.length === 0}>
        </lit-fused-button-bar>
        ${
        this.segmentationMode === SegmentationMode.CUSTOM ? customRegexEntry :
                                                            null}
      </div>
      <div class="controls-group" style="gap: 8px;">
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
        <mwc-icon class='icon-button large-icon' title='Vertical density'
            @click=${onClickToggleVDense}>
            ${this.vDense ? 'expand' : 'compress'}
        </mwc-icon>
        <mwc-icon class='icon-button large-icon'
            title=${
        this.underline ? 'Switch to highlight mode' :
                         'Switch to underline mode'}
            @click=${onClickToggleUnderline}>
            ${this.underline ? 'font_download' : 'format_color_text'}
        </mwc-icon>
      </div>
      <div class='flex-grow-spacer'></div>
    `;
  }

  /* Disabled for space reasons. */
  // renderSelfScoreSelector() {
  //   const onClickToggleSelfSalience = () => {
  //     this.showSelfSalience = !this.showSelfSalience;
  //   };
  //   // prettier-ignore
  //   return html`
  //     <lit-switch labelLeft="Show self scores"
  //       ?selected=${this.showSelfSalience}
  //       @change=${onClickToggleSelfSalience}>
  //     </lit-switch>
  //   `;
  // }
  renderSelfScoreSelector() {
    return null;
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
        <label class="dropdown-label" id="method-label"
         for="method-selector">Method:</label>
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

  renderSalienceTargetStringIndicator() {
    const target = this.salienceTargetOption !== undefined ?
        this.salienceTargetOptions[this.salienceTargetOption] :
        null;
    let sourceInfo = '';
    let targetText = 'none selected.';
    if (target != null) {
      sourceInfo = target.source === TargetSource.REFERENCE ? ' (target)' :
                                                              ' (response)';
      targetText = target.text;
    }

    const targetSelectorHelp =
        'Select a (response) from the model or a pre-defined (target) sequence from the dataset.';

    const isLoadingPreds = this.latestLoadPromises.has('modelPreds');

    const indicatorTextClass = classMap({
      'target-info-line': true,
      'gray-text': target == null,
    });

    const clearSalienceTarget = () => {
      /* this will show the interstitial */
      this.salienceTargetOption = undefined;
    };

    // prettier-ignore
    return html`
      <div class="controls-group controls-group-variable">
        <label class="dropdown-label" title=${targetSelectorHelp}>
          Sequence${sourceInfo}:
        </label>
        <div class=${indicatorTextClass}
         title=${target == null ? 'No target; select one below.' : targetText}>
          ${targetText}
          ${isLoadingPreds ? this.renderLoadingIndicator() : null}
        </div>
      </div>
      <div class='controls-group'>
        <lit-tooltip content=${targetSelectorHelp} tooltipPosition="left"
          id='change-target-button'>
          <button class='hairline-button'
            slot='tooltip-anchor' @click=${clearSalienceTarget}
            ?disabled=${target == null}>
            <span>Select sequence </span><span class='material-icon'>arrow_drop_down</span>
          </button>
        </lit-tooltip>
        <lit-tooltip content=${targetSelectorHelp} tooltipPosition="left"
          id='change-target-icon'>
          <mwc-icon class='icon-button'
            slot='tooltip-anchor' @click=${clearSalienceTarget}>
            edit
          </mwc-icon>
        </lit-tooltip>
      </div>
    `;
  }

  renderLoadingIndicator() {
    // prettier-ignore
    return html`
      <div class='loading-indicator-container'>
        <div class='loading-indicator'></div>
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

    const requestPending = this.targetTokenSpan !== undefined &&
        this.salienceResultCache[this.spanToKey(this.targetTokenSpan)] ===
            REQUEST_PENDING;
    const infoLineClasses = classMap({
      'target-info-line': true,
      'gray-text': requestPending,
    });

    // prettier-ignore
    return html`
      <div class="controls-group controls-group-variable"
        title="Selected target span.">
        <div class=${infoLineClasses}>
          ${printSelectedTargets()}
          ${requestPending ? this.renderLoadingIndicator() : null}
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

  renderTargetSelectorInterstitial() {
    const formatOption = (target: TargetOption, i: number) => {
      const onClickTarget = () => {
        this.salienceTargetOption = i;
      };
      // prettier-ignore
      return html`
        <div class='interstitial-target-option' @click=${onClickTarget}>
          <div class='interstitial-target-text'>${target.text}</div>
        </div>`;
    };

    // Slightly awkward, but we need to process and /then/ filter, because
    // the @click handler needs the original list index.
    const optionsFromDataset =
        this.salienceTargetOptions
            .map((target, i) => {
              if (target.source !== TargetSource.REFERENCE) return null;
              return formatOption(target, i);
            })
            .filter(val => val != null);
    const optionsFromModel =
        this.salienceTargetOptions
            .map((target, i) => {
              if (target.source !== TargetSource.MODEL_OUTPUT) return null;
              return formatOption(target, i);
            })
            .filter(val => val != null);

    const isLoadingPreds = this.latestLoadPromises.has('modelPreds');

    // TODO(b/324959547): get field names 'target' and 'response' from spec
    // via generated_text_utils.ts, rather than hard-coding.
    // This information is available on the frontend, but we need to thread
    // it through a few layers of code in generated_text_utils.ts

    // prettier-ignore
    return html`
      <div class='interstitial-container'>
        <div class='interstitial-contents'>
          <div class='interstitial-header'>
            Choose a sequence to explain
          </div>
          <div class='interstitial-target-selector'>
            <div class='interstitial-target-type'>From dataset (target):</div>
            ${optionsFromDataset}
            <div class='interstitial-target-type'>From model (response):</div>
            ${isLoadingPreds ? this.renderLoadingIndicator() : null}
            ${optionsFromModel}
          </div>
        </div>
      </div>`;
  }

  renderNoExampleInterstitial() {
    // prettier-ignore
    return html`
      <lit-interstitial headline="Sequence Salience">
        Enter a prompt in the Editor or select an example from the Data Table to begin.
      </lit-interstitial>`;
  }

  renderContent() {
    if (this.currentData == null) {
      return this.renderNoExampleInterstitial();
    }

    if (this.salienceTargetOption === undefined) {
      return this.renderTargetSelectorInterstitial();
    }

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

    // prettier-ignore
    return html`
      <div class='chip-container'>
        <lm-salience-chips
          .tokensWithWeights=${segmentsWithWeights} .cmap=${this.cmap}
          ?dense=${this.denseView} ?vDense=${this.vDense}
          ?underline=${this.underline}
          ?preSpace=${this.denseView} breakNewlines>
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
    const onChangeRange = (e: Event) => {
      // Note: HTMLInputElement.valueAsNumber does not work properly for
      // <lit-numeric-input>
      this.cmapRange = Number((e.target as LitNumericInput).value);
    };

    const resetRange = () => {
      this.cmapRange = CMAP_DEFAULT_RANGE;
    };

    // prettier-ignore
    return html`
      <div class="controls-group">
        ${this.renderColorLegend()}
        <mwc-icon class='icon'>opacity</mwc-icon>
        <label id='colormap-slider-label' class='dropdown-label'
         for="cmap-range-slider">Colormap intensity:</label>
        <lit-numeric-input min="0" max="1" step="0.1" id='cmap-range-slider'
          value="${this.cmapRange}" @change=${onChangeRange}>
        </lit-numeric-input>
        <mwc-icon class='icon-button value-reset-icon' title='Reset colormap'
          @click=${resetRange}>
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
          ${this.renderSalienceTargetStringIndicator()}
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