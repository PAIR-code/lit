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
import * as d3 from 'd3';  // Used for computing quantile, not visualization.
import {customElement, html, property} from 'lit-element';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable, when} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {Input, LitName, ModelsMap, SpanLabel, Spec} from '../lib/types';
import {handleEnterKey, isLitSubtype} from '../lib/utils';
import {GroupService} from '../services/group_service';

import {styles} from './datapoint_editor_module.css';
import {styles as sharedStyles} from './shared_styles.css';

// Converter function for text input. Use to support non-string types,
// such as numeric fields.
type InputConverterFn = (s: string) => string|number|string[];

/**
 * A LIT module that allows the user to view and edit a datapoint.
 */
@customElement('datapoint-editor-module')
export class DatapointEditorModule extends LitModule {
  static title = 'Datapoint Editor';
  static numCols = 3;
  static template = (model = '', selectionServiceIndex = 0) => {
    return html`<datapoint-editor-module selectionServiceIndex=${
        selectionServiceIndex}></datapoint-editor-module>`;
  };

  static duplicateForExampleComparison = true;
  static duplicateForModelComparison = false;

  private readonly groupService = app.getService(GroupService);

  static get styles() {
    return [sharedStyles, styles];
  }

  private resizeObserver!: ResizeObserver;

  @observable editedData: Input = {};
  @observable datapointEdited: boolean = false;
  @observable inputHeights: {[name: string]: string} = {};

  @computed
  get dataTextLengths(): {[key: string]: number} {
    const defaultLengths: {[key: string]: number} = {};

    // The string length at this percentile in the input data sample is used
    // to determine the default height of the input box.
    const percentileForDefault = 0.8;

    // Get non-categorical keys.
    const keys = this.appState.currentInputDataKeys.filter((key) => {
      return !(
          this.groupService.categoricalFeatureNames.includes(key) ||
          this.groupService.numericalFeatureNames.includes(key));
    });
    // Get input string lengths for non-categorical keys.
    for (const key of keys) {
      const lengths = this.appState.currentInputData.map(
          indexedInput => indexedInput.data[key]?.length);
      const defaultLength = d3.quantile(lengths, percentileForDefault);
      if (defaultLength == null) continue;

      defaultLengths[key] = defaultLength;
    }

    return defaultLengths;
  }

  firstUpdated() {
    const container = this.shadowRoot!.getElementById('edit-table')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
    this.resizeObserver.observe(container);


    const getCurrentDataset = () => this.appState.currentDataset;
    this.reactImmediately(getCurrentDataset, () => {
      when(() => this.appState.currentInputDataIsLoaded, () => {
        this.resize();
      });
    });

    const getSelectedData = () =>
        this.selectionService.primarySelectedInputData;
    this.reactImmediately(getSelectedData, selectedData => {
      this.resetEditedData(selectedData == null ? null : selectedData.data);
    });
  }

  updated() {
    // Hack to fix the fact that just updating the innerhtml of the dom doesn't
    // update the displayed value of textareas. See
    // https://github.com/elm/virtual-dom/issues/115#issuecomment-329405689.
    const textareas = Array.from(this.shadowRoot!.querySelectorAll('textarea'));
    for (const textarea of textareas) {
      textarea.value = textarea.textContent!;
    }
  }

  private resize() {
    // Get the first input box element, used for  getting the client width to
    // compute the input box height below.
    const inputBoxElement =
        this.shadowRoot!.querySelectorAll('.input-box')[0] as HTMLElement;
    if (inputBoxElement == null) return;

    const keys = Array.from(Object.keys(this.dataTextLengths));
    for (const key of keys) {
      const defaultCharLength = this.dataTextLengths[key];

      // Heuristic for computing height.
      const characterWidth = 13;  // estimate for character width in pixels
      const numLines = Math.ceil(
          characterWidth * defaultCharLength / inputBoxElement.clientWidth);

      // Set 2 ex per line.
      this.inputHeights[key] = `${2 * numLines}ex`;
    }
  }

  // TODO(lit-dev): move to utils or types.ts?
  private defaultValueByField(key: string) {
    const fieldSpec = this.appState.currentDatasetSpec[key];
    if (isLitSubtype(fieldSpec, 'Scalar')) {
      return 0;
    }
    const listFieldTypes: LitName[] =
        ['Tokens', 'SequenceTags', 'SpanLabels', 'EdgeLabels'];
    if (isLitSubtype(fieldSpec, listFieldTypes)) {
      return [];
    }
    const stringFieldTypes: LitName[] = ['TextSegment', 'CategoryLabel'];
    if (isLitSubtype(fieldSpec, stringFieldTypes)) {
      return '';
    }
    console.log(
        'Warning: default value requested for unrecognized input field type',
        key, fieldSpec);
    return '';
  }


  private resetEditedData(selectedInputData: Input|null) {
    this.datapointEdited = false;
    const data: Input = {};

    // If no datapoint is selected, then show an empty datapoint to fill in.
    const keys = selectedInputData == null ?
        this.appState.currentInputDataKeys :
        Object.keys(selectedInputData);
    for (const key of keys) {
      data[key] = selectedInputData == null ? this.defaultValueByField(key) :
                                              selectedInputData[key];
    }
    this.editedData = data;
  }

  render() {
    return html`
      <div id="container">
        ${this.renderEditText()}
        ${this.renderMakeResetButtons()}
      </div>
    `;
  }

  /**
   * Returns false if any of the required model inputs are blank.
   */
  @computed
  private get allRequiredInputsFilledOut() {
    const keys = Object.keys(this.editedData);
    let allRequiredInputsFilledOut = true;
    for (let i = 0; i < keys.length; i++) {
      if (this.appState.currentModelRequiredInputSpecKeys.includes(keys[i]) &&
          this.editedData[keys[i]] === '') {
        allRequiredInputsFilledOut = false;
        break;
      }
    }
    return allRequiredInputsFilledOut;
  }

  renderMakeResetButtons() {
    const makeEnabled = this.datapointEdited && this.allRequiredInputsFilledOut;
    const resetEnabled = this.datapointEdited;

    const onClickNew = async () => {
      const toCreate = [[this.editedData]];
      const ids = [this.selectionService.primarySelectedId!];
      const datapoints =
          await this.appState.createNewDatapoints(toCreate, ids, 'manual', '(manual)');
      this.selectionService.selectIds(datapoints.map(d => d.id));
    };
    const onClickReset = () => {
      this.resetEditedData(
          this.selectionService.primarySelectedInputData!.data);
    };
    return html`
      <div class="button-holder">
        <button id="make"  @click=${onClickNew} ?disabled="${!makeEnabled}">
          Make new Datapoint
        </button>
        <button id="reset" @click=${onClickReset}  ?disabled="${!resetEnabled}">
          Reset
        </button>
      </div>
    `;
  }

  renderEditText() {
    const keys = Object.keys(this.editedData);
    const editable = true;
    // clang-format off
    return html`
      <div id="edit-table">
       ${keys.map(
            (key) => this.renderEntry(key, this.editedData[key], editable))}
      </div>
    `;
    // clang-format on
  }

  // tslint:disable-next-line:no-any
  renderEntry(key: string, value: any, editable: boolean) {
    const handleInputChange =
        (e: Event, converterFn: InputConverterFn = (s => s)) => {
          this.datapointEdited = true;
          // tslint:disable-next-line:no-any
          this.editedData[key] = converterFn((e as any).target.value as string);
        };

    // For categorical outputs, render a dropdown.
    const renderCategoricalInput = (catVals: string[]) => {
      // Note that the first option is blank (so that the dropdown is blank when
      // no point is selected), and disabled (so that datapoints can only have
      // valid values).
      return html`
      <select class="dropdown"
        @change=${handleInputChange}>
        <option value="" selected></option>
        ${catVals.map(val => {
        return html`
            <option
              value="${val}"
              ?selected=${val === value}
              >
              ${val}
            </option>`;
      })}
      </select>`;
    };

    const inputStyle = {'height': this.inputHeights[key]};
    // Render a multi-line text input.
    const renderFreeformInput = () => {
      return html`
      <textarea class="input-box" style="${styleMap(inputStyle)}" @input=${
          handleInputChange}
        ?readonly="${!editable}">${value}</textarea>`;
    };

    // Render a single-line text input.
    const renderShortformInput = () => {
      return html`
      <input type="text" class="input-short" @input=${handleInputChange}
        ?readonly="${!editable}" .value=${value}></input>`;
    };

    // Render a single-line text input, and convert entered value to a number.
    const renderNumericInput = () => {
      const handleNumberInput = (e: Event) => {
        handleInputChange(e, (value: string) => +(value));
      };
      return html`
      <input type="text" class="input-short" @input=${handleNumberInput}
        ?readonly="${!editable}" .value=${value}></input>`;
    };

    // Render tokens as space-separated, but re-split for editing.
    const renderTokensInput = () => {
      const handleTokensInput = (e: Event) => {
        handleInputChange(e, (value: string): string[] => {
          // If value is empty, return [] instead of ['']
          return value ? value.split(' ') : [];
        });
      };
      const valueAsString = value ? value.join(' ') : '';
      return html`
      <textarea class="input-box" style="${styleMap(inputStyle)}" @input=${
          handleTokensInput}
        ?readonly="${!editable}">${valueAsString}</textarea>`;
    };

    // Non-editable render for span labels.
    const renderSpanLabelsNonEditable = () => {
      const renderLabel = (d: SpanLabel) => html`<div class="span-label">[${
          d.start}, ${d.end}): ${d.label}</div>`;
      return html`${value ? (value as SpanLabel[]).map(renderLabel) : null}`;
    };

    let renderInput = renderFreeformInput;  // default: free text
    const fieldSpec = this.appState.currentDatasetSpec[key];
    const vocab = fieldSpec?.vocab;
    if (vocab != null) {
      renderInput = () => renderCategoricalInput(vocab);
    } else if (this.groupService.categoricalFeatureNames.includes(key)) {
      renderInput = renderShortformInput;
    } else if (this.groupService.numericalFeatureNames.includes(key)) {
      renderInput = renderNumericInput;
    } else if (isLitSubtype(fieldSpec, ['Tokens', 'SequenceTags'])) {
      renderInput = renderTokensInput;
    } else if (isLitSubtype(fieldSpec, 'SpanLabels')) {
      renderInput = renderSpanLabelsNonEditable;
    }

    const onKeyUp = (e: KeyboardEvent) => {
      handleEnterKey(e, () => {
        this.shadowRoot!.getElementById('make')!.click();
      });
    };

    // TODO(lit-team): Have better indication of model inputs vs other fields.
    // TODO(lit-team): Should we also display optional input fields that aren't
    // in the dataset? b/157985221.
    const isRequiredModelInput =
        this.appState.currentModelRequiredInputSpecKeys.includes(key);
    const displayKey = `${key}${isRequiredModelInput ? '(*)' : ''}`;
    // Note the "." before "value" in the template below - this is to ensure
    // the value gets set by the template.
    // clang-format off
    return html`
      <div class="entry" @keyup=${(e: KeyboardEvent) => {onKeyUp(e);}}>
        <div><label>${displayKey}: </label></div>
        <div>
          ${renderInput()}
        </div>
      </div>
    `;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'datapoint-editor-module': DatapointEditorModule;
  }
}
