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

import '../elements/checkbox';

// tslint:disable:no-new-decorators
import * as d3 from 'd3';  // Used for computing quantile, not visualization.
import {html} from 'lit';
import {customElement} from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';
import {computed, observable, when} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {AnnotationCluster, EdgeLabel, SpanLabel} from '../lib/dtypes';
import {BooleanLitType, EdgeLabels, Embeddings, ImageBytes, ListLitType, LitTypeWithVocab, MultiSegmentAnnotations, Scalar, SearchQuery, SequenceTags, SpanLabels, SparseMultilabel, StringLitType, Tokens, URLLitType} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatAnnotationCluster, formatEdgeLabel, formatSpanLabel, type IndexedInput, type Input, ModelInfoMap, SCROLL_SYNC_CSS_CLASS, Spec} from '../lib/types';
import {findSpecKeys, isLitSubtype, makeModifiedInput} from '../lib/utils';
import {GroupService} from '../services/group_service';
import {SelectionService} from '../services/selection_service';

import {styles} from './datapoint_editor_module.css';

// Converter function for text input. Use to support non-string types,
// such as numeric fields.
type InputConverterFn = (s: string) => string|number|string[]|boolean|undefined;

/**
 * A LIT module that allows the user to view and edit a datapoint.
 */
@customElement('datapoint-editor-module')
export class DatapointEditorModule extends LitModule {
  static override title = 'Datapoint Editor';
  static override numCols = 2;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
      <datapoint-editor-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </datapoint-editor-module>`;

  static override duplicateForExampleComparison = true;
  static override duplicateForModelComparison = false;

  private readonly groupService = app.getService(GroupService);

  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly resizeObserver = new ResizeObserver(() => {
    this.resize();
  });

  private isShiftPressed = false; /** Newline edits are shift + enter */

  protected addButtonText = 'Add';
  protected showAddAndCompare = true;

  @computed
  get baseData(): IndexedInput {
    const input = this.selectionService.primarySelectedInputData;
    if (input) return input;

    const blankInput = this.appState.makeEmptyDatapoint('manual');
    // Set all numeric fields with out of bounds defaults to an empty string
    // to ensure placeholder value range will render.
    const edits: {[key: string]: string} = {};
    for (const key of this.appState.currentInputDataKeys) {
      if (this.groupService.numericalFeatureNames.includes(key)) {
        const value = blankInput.data[key];
        const fieldSpec = this.appState.currentDatasetSpec[key] as Scalar;
        const [minVal, maxVal] = [fieldSpec.min_val, fieldSpec.max_val];
        if (value < minVal || value > maxVal) {
          edits[key] = "";
        }
      }
    }

    return makeModifiedInput(blankInput, edits, 'manual');
  }

  @observable dataEdits: Input = {};

  @computed
  get missingFields(): string[] {
    // Check only for initial values that are null or undefined for numeric
    // values to allow for defaults that evaluate to false.
    return this.appState.currentModelRequiredInputSpecKeys.filter(
        (key: string) => {
          if (this.groupService.numericalFeatureNames.includes(key)) {
            return this.editedData.data[key] === undefined ||
                this.editedData.data[key] === null;
          }
          else {
            return !this.editedData.data[key];
          }
        });
  }

  @computed
  get invalidNumericFields(): string[] {
    return Object.keys(this.numericFieldRanges).filter((key: string) => {
      const value = this.editedData.data[key];
      if (value === undefined) return true;
      const [minVal, maxVal] = this.numericFieldRanges[key];
      return value < minVal || value > maxVal;
    });
  }

  @computed
  get numericFieldRanges(): {[key: string]: number[]} {
    const ranges: {[key: string]: number[]} = {};
    for (const key of this.appState.currentInputDataKeys) {
      if (this.groupService.numericalFeatureNames.includes(key)) {
        const fieldSpec = this.appState.currentDatasetSpec[key] as Scalar;
        ranges[key] = [fieldSpec.min_val, fieldSpec.max_val];
      }
    }
    return ranges;
  }

  @computed
  get editedData(): IndexedInput {
    return makeModifiedInput(this.baseData, this.dataEdits, 'manual');
  }

  @computed
  get datapointEdited(): boolean {
    return Object.keys(this.dataEdits).length > 0;
  }

  @observable inputHeights: {[name: string]: string} = {};
  @observable maximizedImageFields = new Set<string>();
  @observable editingTokenIndex = -1;
  @observable editingTokenField?: string;
  @observable editingTokenWidth = 0;

  @computed
  get dataTextKeys(): string[]{
   // Returns keys for data text fields.
   const dataTextKeys: string[] = [];

   for (const key of this.appState.currentInputDataKeys) {
     // Skip numerical and categorical keys.
     if (this.groupService.categoricalFeatureNames.includes(key)) continue;
     if (this.groupService.numericalFeatureNames.includes(key)) continue;

     // Skip fields with value type string[] or number[]
     const fieldSpec = this.appState.currentDatasetSpec[key];
     if (fieldSpec instanceof ListLitType) continue;
     if (fieldSpec instanceof Embeddings) continue;

     // Skip image fields
     if (fieldSpec instanceof ImageBytes) continue;

     // Skip boolean fields
     if (fieldSpec instanceof BooleanLitType) continue;

     dataTextKeys.push(key);
    }
    return dataTextKeys;
  }

  @computed
  get sparseMultilabelInputKeys(): string[] {
    const spec = this.appState.currentDatasetSpec;
    return findSpecKeys(spec, SparseMultilabel);
  }

  private calculateQuantileLengthsForFields(
      fieldKeys: string[], percentile = .8) {
    const defaultLengths: {[key: string]: number} = {};

    for (const key of fieldKeys) {
      const fieldSpec = this.appState.currentDatasetSpec[key];
      let calculateStringLength: ((s: string) => number)|
          ((s: string[]) => number);

      if (fieldSpec instanceof StringLitType) {
        calculateStringLength = (s: string) => s.length;
      }
      else if (fieldSpec instanceof SparseMultilabel) {
        const separator = fieldSpec.separator;
        calculateStringLength = (s: string[]) =>
          Object.values(s).join(separator).length;
      }
      else {
        throw new Error(`Attempted to convert field ${
            key} of unrecognized type to string.`);
      }

      const lengths = this.appState.currentInputData.map(indexedInput => {
        return calculateStringLength(indexedInput.data[key]);
      });

      defaultLengths[key] = d3.quantile(lengths, percentile) ?? 1;
      // Override if the distribution is short-tailed, we can expand a bit to
      // avoid scrolling at all. This is useful if everything in a particular
      // column is close to the same length.
      const maxLength = Math.max(...lengths);
      if (percentile * maxLength <= defaultLengths[key]) {
        defaultLengths[key] = maxLength;
      }
    }
    return defaultLengths;
  }

  @computed
  get dataTextLengths(): {[key: string]: number} {
    return this.calculateQuantileLengthsForFields(
      this.dataTextKeys);
  }

  @computed
  get sparseMultilabelInputLengths(): {[key: string]: number} {
    return this.calculateQuantileLengthsForFields(
      this.sparseMultilabelInputKeys);
  }

  override connectedCallback() {
    super.connectedCallback();
    this.reactImmediately(
        () => this.appState.currentDataset,
        () => {
          when(() => this.appState.currentInputDataIsLoaded,
               () => {this.resize();});
        });

    this.react(
        () => this.selectionService.primarySelectedInputData,
        () => {this.resetEditedData();});
  }

  override updated() {
    super.updated();

    // Resize observer only if element is rendered.
    // TODO(lit-dev): consider moving this logic to the LitModule base class,
    // where the decision to render or not is made.
    const container = this.shadowRoot!.querySelector('.module-container')!;
    if (container != null) {
      this.resizeObserver.observe(container);
    } else {
      this.resizeObserver.disconnect();
    }

    // Hack to fix the fact that just updating the innerhtml of the dom doesn't
    // update the displayed value of textareas. See
    // https://github.com/elm/virtual-dom/issues/115#issuecomment-329405689.
    const textareas = Array.from(this.shadowRoot!.querySelectorAll('textarea'));
    for (const textarea of textareas) {
      textarea.value = textarea.textContent!;
    }
  }

  private convertNumLinesToHeight(numLines: number) {
    // Returns a height value in ex.
    const padding = .2;
    return 2.4 * numLines + padding;
  }

  private getHeightForInputBox(
      textLength: number, clientWidth: number,
      convertToString = true) {
    const characterWidth = 8.3;  // estimate for character width in pixels
    const numLines = Math.ceil(characterWidth * textLength / clientWidth);
    const height = this.convertNumLinesToHeight(Math.max(1, numLines));
    return convertToString ? `${height}ex` : height;
  }

  private resize() {
    // Get the first input box element, used for  getting the client width to
    // compute the input box height below.
    const inputBoxElement =
        this.shadowRoot!.querySelectorAll('.input-box')[0] as HTMLElement;
    if (inputBoxElement == null) return;

    // Set heights for string-based input boxes.
    for (const [key, defaultCharLength] of
             Object.entries(this.dataTextLengths)) {
      if (defaultCharLength === -Infinity) {
        continue;
      }

      this.inputHeights[key] =
          this.getHeightForInputBox(
              defaultCharLength, inputBoxElement.clientWidth) as string;
    }

    // Set heights for multi-label input boxes.
    for (const [key, textLength] of
             Object.entries(this.sparseMultilabelInputLengths)) {

      // Truncate input height to 4 lines maximum.
      const maxHeight = this.convertNumLinesToHeight(4);
      const defaultHeight =
          this.getHeightForInputBox(
              textLength, inputBoxElement.clientWidth, false) as number;
      const height = Math.min(defaultHeight, maxHeight);
      this.inputHeights[key] = `${height}ex`;
    }
  }

  private resetEditedData() {
    this.editingTokenIndex = -1;
    this.editingTokenField = undefined;
    this.dataEdits = {};
  }

  override renderImpl() {
    // Scrolling inside this module is done inside a div with ID 'container'.
    // Giving this div the class defined by SCROLL_SYNC_CSS_CLASS allows
    // scrolling to be sync'd instances of this module when doing comparisons
    // between models and/or duplicated datapoints. See lit_module.ts for more
    // details.
    return html`
      <div class='module-container'>
        <div class="${SCROLL_SYNC_CSS_CLASS} module-results-area">
          ${this.renderEditableFields()}
        </div>
        <div class="module-footer">
          ${this.renderButtons()}
        </div>
      </div>
    `;
  }

  renderButtons() {
    const hasRequiredFields = this.missingFields.length === 0;
    const hasNoInvalidValues = this.invalidNumericFields.length === 0;
    const makeEnabled =
        this.datapointEdited && hasRequiredFields && hasNoInvalidValues;
    const compareEnabled = makeEnabled && !this.appState.compareExamplesEnabled;
    const resetEnabled = this.datapointEdited;
    const clearEnabled = this.selectionService.primarySelectedInputData != null;

    const onClickNew = async () => {
      const data: IndexedInput[] =
          await this.appState.annotateNewData([this.editedData]);
      this.appState.commitNewDatapoints(data);
      const newIds = data.map(d => d.id);
      this.selectionService.selectIds(newIds);
      return newIds;
    };
    const onClickCompare = async () => {
      const parentId = this.selectionService.primarySelectedId!;
      const newIds = await onClickNew();
      this.appState.compareExamplesEnabled = true;
      // Explicitly set the reference to be the original parent example and
      // the primary selection to the new example.
      app.getService(SelectionService, 'pinned').selectIds([parentId]);
      app.getService(SelectionService).selectIds(newIds);
    };
    const onClickReset = () => {
      this.resetEditedData();
    };
    const onClickClear = () => {
      this.selectionService.selectIds([]);
    };

    const analyzeButton = html`
      <button id="make" class='hairline-button'
        @click=${onClickNew}
        ?disabled="${!makeEnabled}">
        ${this.addButtonText}
      </button>
    `;
    const compareButton = html`
      <button id="compare" class='hairline-button' style="text-wrap: nowrap;"
        @click=${onClickCompare} ?disabled="${!compareEnabled}">
        Add and compare
      </button>
    `;
    const resetButton = html`
      <button id="reset" class='hairline-button'
        @click=${onClickReset}  ?disabled="${!resetEnabled}">
        Reset
      </button>
    `;
    const clearButton = html`
      <button id="clear" class='hairline-button'
        @click=${onClickClear}  ?disabled="${!clearEnabled}">
        Clear
      </button>
    `;

    // clang-format off
    return html`
      ${analyzeButton}
      ${this.showAddAndCompare ? compareButton : null}
      ${resetButton}
      ${clearButton}
    `;
    // clang-format on
  }

  renderEditableFields() {
    const keys = Object.keys(this.appState.currentDatasetSpec);
    // clang-format off
    return html`
      <div id="edit-table">
       ${keys.map(
            key => this.renderEntry(key, this.editedData.data[key]))}
      </div>
    `;
    // clang-format on
  }

  // tslint:disable-next-line:no-any
  renderEntry(key: string, value: any) {
    const handleInputChange =
        (e: Event, converterFn: InputConverterFn = (s => s)) => {
          // tslint:disable-next-line:no-any
          const value = converterFn((e as any).target.value as string);
          if (value === this.baseData.data[key]) {
            delete this.dataEdits[key];
          } else {
            this.dataEdits[key] = value;
          }
        };

    // For categorical fields, render a dropdown.
    const renderCategoricalInput = (catVals: string[]) => {
      // clang-format off
      return html`
        <select class="dropdown"
          @change=${handleInputChange} .value=${value}>
          ${catVals.map(val => html`
            <option value="${val}" ?selected=${val === value}>${val}</option>
          `)}
        </select>
      `;
      // clang-format on
    };

    // Render an image.
    const renderImage = () => {
      const toggleImageSize = () => {
        if (this.maximizedImageFields.has(key)) {
          this.maximizedImageFields.delete(key);
        } else {
          this.maximizedImageFields.add(key);
        }
      };
      const uploadClicked = () => {
        this.shadowRoot!.querySelector<HTMLInputElement>('#uploadimage')!.click();
      };
      const handleUpload = (e: Event) => {
        const inputElem = e.target as HTMLInputElement;
        const file = inputElem.files == null || inputElem.files.length === 0 ?
            null :
            inputElem.files[0];
        if (file == null) {
          return;
        }
        const reader = new FileReader();
        reader.addEventListener('load', () => {
          const value = reader.result as string;
          if (value === this.baseData.data[key]) {
            delete this.dataEdits[key];
          } else {
            this.dataEdits[key] = value;
          }
        });
        reader.readAsDataURL(file);
        inputElem.value = '';
      };
      const maximizeImage = this.maximizedImageFields.has(key);
      const imageSource = (value == null) ? '' : value.toString() as string;
      const noImage = imageSource === '';
      const imageClasses =
          classMap({'image-min': !maximizeImage, 'hidden': noImage});
      const toggleClasses = classMap({'image-toggle': true, 'hidden': noImage});
      const uploadLabel = noImage ? 'Upload image' : 'Replace image';
      return html`
        <div class="image-holder">
          <img class=${imageClasses} src=${imageSource}>
          <mwc-icon class="${toggleClasses}" @click=${toggleImageSize}
                    title="Toggle full size">
            ${maximizeImage ? 'photo_size_select_small' :
              'photo_size_select_large'}
          </mwc-icon>
          <div>
            <input type='file' id='uploadimage' accept="image/*"
                   @change=${handleUpload} class="hidden">
            <button class="hairline-button image-button"
                   @click=${uploadClicked}>
              <span class="material-icon">publish</span>
              ${uploadLabel}
            </button>
          </div>
        </div>
      `;
    };

    // TODO(lit-team): Have better indication of model inputs vs other fields.
    // TODO(lit-team): Should we also display optional input fields that aren't
    // in the dataset? b/157985221.
    const isRequiredModelInput =
        this.appState.currentModelRequiredInputSpecKeys.includes(key);

    const isMissingField = this.missingFields.includes(key);
    const isInvalidNumber = this.invalidNumericFields.includes(key);
    const renderError =
        (isMissingField && isRequiredModelInput || isInvalidNumber) &&
        this.datapointEdited;

    const errorInputClasses = renderError ? 'error-input' : '';
    const errorIconClasses = renderError ? 'error-icon' : '';

    const inputStyle = {'min-height': this.inputHeights[key]};
    // Render a multi-line text input.
    const renderFreeformInput = () => html`
      <textarea
        class="input-box ${errorInputClasses}"
        style="${styleMap(inputStyle)}"
        @input=${handleInputChange}>${value}</textarea>`;

    // Render a single-line text input.
    const renderShortformInput = () =>
        html`<input type="text" class="input-short ${errorInputClasses}"
          @input=${handleInputChange} .value=${value} />`;

    // Render a single-line text input, and convert entered value to a number.
    const renderNumericInput = () => {
      const handleNumberInput = (e: Event) => {
        handleInputChange(e, (value: string | undefined) => {
          // Treat empty input as invalid, rather than coerce to zero.
          if (value === "") return undefined;
          return Number(value);
        });
      };
      const [minVal, maxVal] = this.numericFieldRanges[key];
      return html`
      <input type="number" class="input-short ${errorInputClasses}"
        @input=${handleNumberInput}
        .placeholder="${minVal} — ${maxVal}"
        .value=${value} />`;
    };

    const renderSpanLabel = (d: SpanLabel) =>
        html`<div class="monospace-label">${formatSpanLabel(d)}</div>`;

    // Non-editable render for span labels.
    const renderSpanLabelsNonEditable =
        () => html`<div>${
          value ? (value as SpanLabel[]).map(renderSpanLabel) : null}</div>`;

          // Non-editable render for edge labels.
    const renderEdgeLabelsNonEditable = () => {
      const renderLabel = (d: EdgeLabel) => {
        return html`<div class="monospace-label">${formatEdgeLabel(d)}</div>`;
      };
      return html`<div>${
          value ? (value as EdgeLabel[]).map(renderLabel) : null}</div>`;
    };
    // Non-editable render for multi-segment annotations.
    const renderMultiSegmentAnnotationsNonEditable = () => {
      const renderLabel = (ac: AnnotationCluster) => {
        return html`<div class="annotation-cluster">
          <div>${formatAnnotationCluster(ac)}</div>
          <ul>${ac.spans.map(s => html`<li>${renderSpanLabel(s)}</li>`)}</ul>
        </div>`;
      };
      return html`<div class="multi-segment-annotation">${
          value ? (value as AnnotationCluster[]).map(renderLabel) : ''}</div>`;
    };

    // Non-editable render for embeddings fields.
    // We technically can use the token editor for these, but it is very
    // unwieldy.
    function renderEmbeddingsNonEditable() {
      // clang-format off
      return html`<div class="monospace-label">
        ${value ? html`&lt;float&gt;[${value.length}]` : null}
      </div>`;
      // clang-format on
    }

    // For boolean values, render a checkbox.
    const renderBoolean = () => {
      const handleCheckboxChange = (e: Event) => {
        // Converter function ignores 'value' input string, uses checked status.
        handleInputChange(e, () => !!(e.target as HTMLInputElement).checked);
      };
      return html`
      <lit-checkbox
        ?checked=${value}
        @change=${handleCheckboxChange}
      ></lit-checkbox>`;
    };

    const renderTokensInput = () => {
      return this.renderTokensInput(
          key, value, handleInputChange,
          !(fieldSpec instanceof Embeddings));
    };

    const entryClasses = {
      'entry': true,
      'entry-edited': this.dataEdits[key] !== undefined,
      'entry-medium': false,
      'entry-long': false,
      'entry-left-align': false,
    };

    let renderInput = renderFreeformInput;  // default: free text
    const fieldSpec = this.appState.currentDatasetSpec[key];
    const {vocab} = fieldSpec as LitTypeWithVocab;
    if (vocab != null && !(fieldSpec instanceof SparseMultilabel)) {
      renderInput = () => renderCategoricalInput(vocab);
    } else if (this.groupService.categoricalFeatureNames.includes(key)) {
      renderInput = renderShortformInput;
      entryClasses['entry-medium'] = true;
    } else if (this.groupService.numericalFeatureNames.includes(key)) {
      renderInput = renderNumericInput;
    } else if (isLitSubtype(
                   fieldSpec, [Tokens, SequenceTags, SparseMultilabel])) {
      renderInput = renderTokensInput;
      entryClasses['entry-long'] = true;
      entryClasses['entry-left-align'] = true;
    } else if (fieldSpec instanceof Embeddings) {
      renderInput = renderEmbeddingsNonEditable;
    } else if (fieldSpec instanceof SpanLabels) {
      renderInput = renderSpanLabelsNonEditable;
    } else if (fieldSpec instanceof EdgeLabels) {
      renderInput = renderEdgeLabelsNonEditable;
    } else if (fieldSpec instanceof MultiSegmentAnnotations) {
      renderInput = renderMultiSegmentAnnotationsNonEditable;
    } else if (fieldSpec instanceof ImageBytes) {
      renderInput = renderImage;
    } else if (fieldSpec instanceof BooleanLitType) {
      renderInput = renderBoolean;
    } else {
      entryClasses['entry-long'] = true;
    }

    // Shift + enter creates a newline; enter alone creates a new datapoint.
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift') this.isShiftPressed = false;
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift') this.isShiftPressed = true;
      if (e.key === 'Enter') {
        if (!this.isShiftPressed) {
          e.preventDefault();
          this.shadowRoot!.getElementById('make')!.click();
        }
      }
    };

    let toolTipContent = '';

    if (isInvalidNumber) {
        const [minVal, maxVal] = this.numericFieldRanges[key];
        toolTipContent = `Please choose a value between ${minVal} and
          ${maxVal}.`;
    } else if (renderError) {
      toolTipContent = `Please enter a valid value for the ${key} field.`;
    }

    let headerContent = html`${isRequiredModelInput ? '*' : ''}${key}`;
    if (fieldSpec instanceof URLLitType) {
      headerContent = html`
        <a href=${value as string} target="_blank">
          ${headerContent}
          <mwc-icon class="icon-button">open_in_new</mwc-icon>
        </a>`;
    } else if (fieldSpec instanceof SearchQuery) {
      const params = new URLSearchParams();
      params.set('q', value);
      headerContent = html`
        <a href="https://www.google.com/search?${
          params.toString()}" target="_blank">
          ${headerContent}
          <mwc-icon class="icon-button google-icon">google</mwc-icon>
        </a>`;
    }

    // Note the "." before "value" in the template below - this is to ensure
    // the value gets set by the template.
    // clang-format off
    return html`
      <div class=${classMap(entryClasses)}
        @keyup=${(e: KeyboardEvent) => {onKeyUp(e);}}
        @keydown=${(e: KeyboardEvent) => {onKeyDown(e);}}
        >
        <div class='field-header'>
          <div class='field-name'>${headerContent}</div>
          <div class='field-type'>${fieldSpec.name}</div>
          <lit-tooltip content=${toolTipContent}>
            <mwc-icon slot="tooltip-anchor" class="${errorIconClasses}">
              ${renderError ? 'error' : null}
            </mwc-icon>
          </lit-tooltip>
        </div>
        <div class='entry-content'>
          ${renderInput()}
        </div>
      </div>
    `;
    // clang-format on
  }

  /**
   * Renders an input value as tokens.
   *
   * Args:
   *   key: Feature name
   *   value: List of values for the feature
   *   handleInputChange: Callback on changes to the feature values
   *   dyanmicTokenLength: If true, allow adding/deleting of tokens.
   */
  renderTokensInput(
      key: string, value: string[],
      handleInputChange: (e: Event, converterFn: InputConverterFn) => void,
      dynamicTokenLength = true) {
    const tokenValues = value == null ? [] : [...value];
    const tokenRenders = [];
    for (let i = 0; i < tokenValues.length; i++) {
      const tokenOrigValue = tokenValues[i];
      const handleTokenInput = (e: Event) => {
        handleInputChange(e, (tokenValue: string): string[] => {
          tokenValues[i] = tokenValue;
          return tokenValues;
        });
      };
      const showTextArea =
          this.editingTokenIndex === i && this.editingTokenField === key;
      const deleteToken = (e: Event) => {
        handleInputChange(e, (): string[] => {
          tokenValues.splice(i, 1);
          return tokenValues;
        });
      };
      const insertToken = (e: Event) => {
        handleInputChange(e, (): string[] => {
          tokenValues.splice(i + 1, 0, '');
          return tokenValues;
        });
        this.editingTokenIndex = i + 1;
        this.editingTokenField = key;
      };
      const renderDeleteButton = () => showTextArea ?
          // clang-format off
            html`<mwc-icon class="icon-button delete-button"
                           title="delete token"
                           @click=${deleteToken}>delete</mwc-icon>` :
          // clang-format on
          null;
      const insertButtonClass = classMap({
        'insert-token-div': true,
      });
      const handleTokenClick = (e: Event) => {
        this.editingTokenIndex = i;
        this.editingTokenField = key;
        this.editingTokenWidth = (e.target as HTMLElement).clientWidth;
      };
      const handleTokenFocusOut = (e: Event) => {
        // Reset our editingTokenIndex after a timeout so as to allow for
        // the delete token button to be pressed, as that also removes focus.
        window.setTimeout(() => {
          if (this.editingTokenIndex === i) {
            this.editingTokenIndex = -1;
          }
        }, 200);
      };
      const renderTextArea = () => {
        requestAnimationFrame(() => {
          const textarea =
              this.renderRoot.querySelector<HTMLElement>('.token-box');
          if (textarea != null) {
            textarea.focus();
          }
        });

        const TEXTAREA_EXTRA_WIDTH = 60;
        const width = this.editingTokenWidth + TEXTAREA_EXTRA_WIDTH;
        const textareaStyle = styleMap({'width': `${width}px`});
        return html`
            <textarea class="token-box"
                      style=${textareaStyle}
                      @input=${handleTokenInput} rows=1
                      @focusout=${handleTokenFocusOut}
            >${tokenOrigValue}</textarea>`;
      };
      const renderDiv = () => html`
        <div class="token-div"
             @click=${handleTokenClick}>${tokenOrigValue}</div>`;
      const renderInsertTokenButton = () => html`
        <div class="${insertButtonClass}" @click=${insertToken}
            title="insert token">
        </div>`;
      tokenRenders.push(
          // clang-format off
          html`<div class="token-outer">
                 <div class="token-holder">
                   ${showTextArea ? renderTextArea() : renderDiv()}
                   ${dynamicTokenLength ? renderDeleteButton() : null}
                 </div>
                 ${dynamicTokenLength ? renderInsertTokenButton() : null}
              </div>`);
      // clang-format on
    }
    const newToken = (e: Event) => {
      handleInputChange(e, (): string[] => {
        tokenValues.push('');
        return tokenValues;
      });
      this.editingTokenIndex = tokenValues.length - 1;
      this.editingTokenField = key;
    };
    const renderAddTokenButton = () =>
        html`<mwc-icon class="icon-button token-button" @click=${newToken}
                  title="insert token">add
           </mwc-icon>`;
    return html`<div class="tokens-holder">
        ${tokenRenders.map(tokenRender => tokenRender)}
        ${dynamicTokenLength ? renderAddTokenButton() : null}
        </div>`;
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

/**
 * Regular datapoint editor, but does not duplicate in example SxS mode.
 */
@customElement('single-datapoint-editor-module')
export class SingleDatapointEditorModule extends DatapointEditorModule {
  static override duplicateForExampleComparison = false;
  static override template = () => {
    return html`<single-datapoint-editor-module></single-datapoint-editor-module>`;
  };
}

/**
 * Simplified version of the above; omits add-and-compare button.
 */
@customElement('simple-datapoint-editor-module')
export class SimpleDatapointEditorModule extends SingleDatapointEditorModule {
  protected override addButtonText = 'Analyze';
  protected override showAddAndCompare = false;
  static override template = () => {
    return html`<simple-datapoint-editor-module></simple-datapoint-editor-module>`;
  };
}

declare global {
  interface HTMLElementTagNameMap {
    'datapoint-editor-module': DatapointEditorModule;
    'single-datapoint-editor-module': SingleDatapointEditorModule;
    'simple-datapoint-editor-module': SimpleDatapointEditorModule;
  }
}
