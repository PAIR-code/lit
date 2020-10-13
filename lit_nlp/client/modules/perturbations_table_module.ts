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
import '@material/mwc-icon';

import * as d3 from 'd3';
import {customElement, html, TemplateResult} from 'lit-element';
import {observable} from 'mobx';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';

import '../elements/text_diff';
import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {LitName, ModelsMap, Spec} from '../lib/types';
import {doesOutputSpecContain, formatLabelNumber, findSpecKeys} from '../lib/utils';
import {DeltasService} from '../services/services';
import {DeltaRow, Source} from '../services/deltas_service';

import {styles} from './perturbations_table_module.css';
import {styles as sharedStyles} from './shared_styles.css';

type DeltaRowsById = {
  [id: string]: DeltaRow
};

type InputSource = {
  modelName: string
  specKey: string
  fieldName: string
};

type TableSortFn = (row: TableData, column: number) => any;

type FormattedTableData = {
  rows: TableData[]
  columns: Map<string, boolean>
  getId: (row: TableData) => string
  sortFn: TableSortFn
}

/**
 * Module to sort generated countefactuals by the change in prediction for a
 * regression or multiclass classification model.
 */
@customElement('perturbations-table-module')
export class PerturbationsTableModule extends LitModule {
  static title = 'Perturbations';
  static numCols = 4;
  static duplicateForModelComparison = true;
  static duplicateAsRow = false;
  static template = (model = '') => {
    return html`<perturbations-table-module model=${model}></perturbations-table-module>`;
  };

  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly deltasService = app.getService(DeltasService);

  @observable private filterSelected = false;
  @observable private lastSelectedSourceIndex?: number;

  /* Enforce datapoint selection */
  private filteredDeltaRows(deltaRows: DeltaRow[]): DeltaRow[] {
    return (this.filterSelected)
      ? this.deltasService.selectedDeltaRows(deltaRows)
      : deltaRows;
  }

  /* Default to the first source if none has been selected yet */
  private selectedSourceIndex(): number {
    return this.lastSelectedSourceIndex ?? 0;
  };


  /* Examples may have various text input fields that are relevant to show */
  private findInputTextFields(): InputSource[] {
    const modelName = this.model;
    const modelSpec = this.appState.getModelSpec(modelName);
    const inputSpecKeys: LitName[] = ['TextSegment', 'GeneratedText'];
    return inputSpecKeys.flatMap(specKey => {
      const fieldNames = findSpecKeys(modelSpec.input, [specKey]);
      return fieldNames.map(fieldName => ({modelName, specKey, fieldName}));
     });
  }
  
  private formatSourceName(source: Source) {
    const {modelName, specKey, fieldName, predictionLabel} = source;
    return (predictionLabel != null)
      ? `${fieldName}:${predictionLabel}`
      : fieldName;
  }

  private onSelect(selectedRowIndices: number[]) {
    const ids = selectedRowIndices
                    .map(index => this.appState.currentInputData[index]?.id)
                    .filter(id => id != null);
    this.selectionService.selectIds(ids);
  }

  private onPrimarySelect(index: number) {
    const id = (index === -1)
      ? null
      : this.appState.currentInputData[index]?.id ?? null;
    this.selectionService.setPrimarySelection(id);
  }

  render() {
    const ds = this.appState.generatedDataPoints;
    if (ds.length === 0) {
      return html`<div class="info">No counterfactuals created yet.</div>`;
    }

    /* Consider classification and regression predictions, and fan out to
     * each (model, outputKey, fieldName, predictionLabel) and show only
     * the selected source.
     */
    const sources = this.deltasService.sourcesForModel(this.model);
    const sourceIndex = this.selectedSourceIndex();
    const source = sources[sourceIndex];
    const navigationStrip = this.renderNavigationStrip(sources);
    return this.renderHeaderAndTable(source, sourceIndex, navigationStrip);
  }

  private renderHeaderAndTable(
    source: Source,
    sourceIndex: number,
    navigationStrip?: TemplateResult
  ) {
    const allDeltaRows = this.deltasService.readDeltaRowsForSource(source);
    const filteredDeltaRows = this.filteredDeltaRows(allDeltaRows);
    const deltaRowsById: DeltaRowsById = {};
    allDeltaRows.forEach(deltaRow => deltaRowsById[deltaRow.d.id] = deltaRow);
    const formattedTableData = this.formattedTable(source, filteredDeltaRows, deltaRowsById);

    const rootClass = classMap({
      'hidden': (sourceIndex !== (this.selectedSourceIndex())),
      [sourceIndex]: true
    });
    return html`
      <div class=${rootClass}>
        ${this.renderHeader(allDeltaRows, navigationStrip)}
        ${this.renderTable(source, formattedTableData)}
      </div>
    `;
  }

  private renderHeader(deltaRows: DeltaRow[], navigationStrip?: TemplateResult) {
    const toggleFilterSelected = () => {
      this.filterSelected = !this.filterSelected;
    };
    const onSelectGenerated = () => {
      const ids = this.appState.generatedDataPoints.map(d => d.id);
      this.selectionService.selectIds(ids);
    };
    return html`
      <div class="info">
        <div class="header-and-actions">
          <div class="header-text">
            Generated ${deltaRows.length === 1 ? '1 datapoint' : `${deltaRows.length} datapoints`}
          </div>
          <lit-checkbox
            label="Only show selected"
            ?checked=${this.filterSelected}
            @change=${toggleFilterSelected}
          ></lit-checkbox>
          <button class="plain" @click=${onSelectGenerated}>Select generated</button>
        </div>
        ${navigationStrip ?? null}
       </div>
    `;
  }

  private renderNavigationStrip(sources: Source[]) {
    if (sources.length === 1) {
      return undefined;
    }

    const onChangeSource = (e: Event) => {
      const select = (e.target as HTMLSelectElement);
      this.lastSelectedSourceIndex = +select.selectedIndex;
    };

    const options = sources.map((source, index) => {
      const text = this.formatSourceName(source);
      return {
        text,
        value: index
      };
    });
    return html`
      <div class="dropdown-holder">
        <label class="dropdown-label">Field</label>
        <select class="dropdown" @change=${onChangeSource} .value=${this.selectedSourceIndex()}>
          ${options.map(option => html`
            <option value=${option.value}>${option.text}</option>
          `)}
        </select>
      </div>
    `;
  }

  private renderSentenceWithDiff(before: string, after: string) {
    return html`
      <lit-text-diff
        beforeText=${before}
        afterText=${after}
        .includeBefore=${false}
      ></lit-text-diff>
    `;
  }

  private renderDeltaCell(delta: number, meanAbsDelta?: number) {
    /// Styles, because classes won't apply within the table's shadow DOM
    const opacity = meanAbsDelta ? Math.abs(delta) / meanAbsDelta : 0.5;
    const styles = styleMap({
      'font-size': '12px',
      'opacity': opacity.toFixed(3),
      'vertical-align': 'middle'
    });
    return html`
      <div>
        <mwc-icon style=${styles}>
          ${delta > 0 ? 'arrow_upward' : 'arrow_downward'}
        </mwc-icon>
        ${formatLabelNumber(Math.abs(delta))}
      </div>
    `;
  }


  /* The definition of the columns, the formatting of the rows, and the sort function
   * are all inherently coupled, so they're isolated and defined jointly in this method */
  private formattedTable(
    source: Source,
    deltaRows: DeltaRow[],
    deltaRowsById: DeltaRowsById
  ): FormattedTableData {
    // Column indexes change, since the number of input fields is variable  
    const ID_COLUMN = 0;
    const COLUMNS_BEFORE_TEXT = 1;
    const DELTA_IS_N_COLUMNS_AFTER_TEXT = 4;
    const inputTextFields = this.findInputTextFields();

    // Define the columns
    const columns = new Map<string, boolean>();
    columns.set('id', false); // hidden, and used for lookups
    inputTextFields.forEach(input => {
      columns.set(`generated ${input.fieldName}`, true);
    });
    columns.set(`source`, true);
    columns.set(`parent ${source.fieldName}`, true);
    columns.set(`${source.fieldName}`, true);
    columns.set('delta', true);

    // Format each row of data
    const BLANK = '-';
    const meanAbsDelta = d3.mean(deltaRows.filter(d => d.delta != null), d => {
      return Math.abs(d.delta!);
    });
    const rows = deltaRows.map((deltaRow: DeltaRow) => {
      const {before, after, delta, d, parent}  = deltaRow;
      const textCells = inputTextFields.map(({fieldName}) => {
        return this.renderSentenceWithDiff(parent.data[fieldName], d.data[fieldName]);
      });
      const {ruleText} = this.deltasService.interpretGenerator(d);

      // When changing the order here, check that references to these constants still
      // make sense throughout the rest of this method.
      return [
        d.id, // ID_COLUMN
        ...textCells,  // COLUMNS_BEFORE_TEXT
        ruleText,
        before ? formatLabelNumber(before) : BLANK,
        after ? formatLabelNumber(after) : BLANK,
        delta ? this.renderDeltaCell(delta, meanAbsDelta) : BLANK // DELTA_IS_N_COLUMNS_AFTER_TEXT
      ];
    });

    // Describe sorting for each column, since some can't naively sort on HTML
    const getId = (row: TableData)=> row[ID_COLUMN] as string;
    const sortFn = (row: TableData, column: number) => {
      const id = getId(row);
      const deltaRow = deltaRowsById[id];

      // input text fields should sort by value, not the rendered HTML diff
      const maybeTextColumnIndex = column - COLUMNS_BEFORE_TEXT;
      if (maybeTextColumnIndex >= 0 && maybeTextColumnIndex < inputTextFields.length) {
        const {fieldName} = inputTextFields[maybeTextColumnIndex];
        return deltaRow.d.data[fieldName];
      }
      
      // deltas should sort numeric, not by rendered HTML with symbols
      const deltaColumnIndex = inputTextFields.length + DELTA_IS_N_COLUMNS_AFTER_TEXT;
      if (column === deltaColumnIndex) {
        return deltaRow.delta;
      }

      return row[column];
    };

    return {rows, columns, getId, sortFn};
  }

  private renderTable(output: Source, tableData: FormattedTableData) {
    const {rows, columns, sortFn, getId} = tableData;

    const primarySelectedIndex =
      this.appState.getIndexById(this.selectionService.primarySelectedId);
    const onSelect = (selectedRowIndices: number[]) => {
      this.onSelect(selectedRowIndices);
    };
    const onPrimarySelect = (index: number) => {
      this.onPrimarySelect(index);
    };
    const getSortValue = (row: TableData, column: number) => {
      return sortFn(row, column)
    };
    const getDataIndexFromRow = (row: TableData) => {
      const id = getId(row);
      return this.appState.getIndexById(id as string);
    };

    return html`
      <div class="table-container">
        <lit-data-table
          class="table"
          defaultSortName="delta"
          .defaultSortAscending=${false}
          .columnVisibility=${columns}
          .data=${rows}
          .selectedIndices=${this.selectionService.selectedRowIndices}
          .primarySelectedIndex=${primarySelectedIndex}
          .onSelect=${onSelect}
          .onPrimarySelect=${onPrimarySelect}
          .getDataIndexFromRow=${getDataIndexFromRow}
          .getSortValue=${getSortValue}
        ></lit-data-table>
      </div>
    `;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return doesOutputSpecContain(modelSpecs, [
      'RegressionScore',
      'MulticlassPreds'
    ]);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'perturbations-table-module': PerturbationsTableModule;
  }
}
