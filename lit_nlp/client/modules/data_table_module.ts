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
import '../elements/checkbox';

import {customElement, html} from 'lit-element';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {LitModule} from '../core/lit_module';
import {TableData} from '../elements/table';
import {IndexedInput, LitType, ModelsMap, SpanLabel, Spec} from '../lib/types';
import {compareArrays, findSpecKeys, isLitSubtype, shortenId} from '../lib/utils';
import {ClassificationInfo} from '../services/classification_service';
import {RegressionInfo} from '../services/regression_service';
import {ClassificationService, RegressionService, SelectionService} from '../services/services';

import {styles} from './data_table_module.css';
import {styles as sharedStyles} from './shared_styles.css';

/**
 * A LIT module showing a table containing the InputData examples. Allows the
 * user to sort, filter, and select examples.
 */
@customElement('data-table-module')
export class DataTableModule extends LitModule {
  static title = 'Data Table';
  static template = () => {
    return html`<data-table-module></data-table-module>`;
  };
  static numCols = 5;
  static get styles() {
    return [sharedStyles, styles];
  }

  static duplicateForModelComparison = false;

  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly regressionService = app.getService(RegressionService);

  @observable columnVisibility = new Map<string, boolean>();
  @observable
  modelPredToClassificationInfo = new Map<string, ClassificationInfo[]>();
  @observable modelPredToRegressionInfo = new Map<string, RegressionInfo[]>();
  @observable searchText = '';
  @observable filterSelected = false;

  @computed
  get dataSpec(): Spec {
    return this.appState.currentDatasetSpec;
  }

  @computed
  get keys(): string[] {
    // Use currentInputData to get keys / column names because filteredData
    // might have 0 length;
    const keys = this.appState.currentInputDataKeys.filter(d => d !== 'meta');
    return keys;
  }

  @computed
  get defaultColumns(): string[] {
    return ['index', 'id', ...this.keys];
  }

  /**
   * Keys of the format 'rowName:columnName' (formatted with getTableKey()) map
   * to the cell value at this row and column location.
   */
  @computed
  get keysToTableEntry() {
    const models = this.appState.currentModels;
    const keysToTableEntry = new Map<string, string|number>();

    // Update the stored table entries map with prediction info.
    models.forEach((model) => {
      const outputSpec = this.appState.currentModelSpecs[model].spec.output;
      const classificationKeys = findSpecKeys(outputSpec, ['MulticlassPreds']);
      const regressionKeys = findSpecKeys(outputSpec, ['RegressionScore']);

      this.addTableEntries(
          keysToTableEntry, model, classificationKeys,
          this.modelPredToClassificationInfo,
          this.classificationService.getDisplayNames());

      this.addTableEntries(
          keysToTableEntry, model, regressionKeys,
          this.modelPredToRegressionInfo,
          this.regressionService.getDisplayNames());
    });

    return keysToTableEntry;
  }

  /**
   * Recursively follow parent pointers and list their numerical indices.
   * Returns a list with the current index last, e.g.
   * [grandparent, parent, child]
   */
  private reversedAncestorIndices(d: IndexedInput): number[] {
    const ancestorIds = this.appState.getAncestry(d.id);
    // Convert to indices and return in reverse order.
    return ancestorIds.map((id) => this.appState.indicesById.get(id)!)
        .reverse();
  }

  // TODO(lit-dev): figure out why this updates so many times;
  // it gets run _four_ times every time a new datapoint is added.
  @computed
  get data(): TableData[] {
    const inputData = this.appState.currentInputData;

    // TODO(lit-dev): pre-compute the index chains for each point, since
    // this might get slow if we have a lot of counterfactuals.
    const sortedData = inputData.slice().sort(
        (a, b) => compareArrays(
            this.reversedAncestorIndices(a), this.reversedAncestorIndices(b)));

    // TODO(b/160170742): Make data table render immediately once the
    // non-prediction data is available, then fetch predictions asynchronously
    // and enable the additional columns when ready.
    return sortedData.map((d) => {
      let displayId = shortenId(d.id);
      displayId = displayId ? displayId + '...' : '';
      // Add an asterisk for generated examples
      // TODO(lit-dev): set up something with CSS instead.
      displayId = d.meta['added'] ? '*' + displayId : displayId;
      const index = this.appState.indicesById.get(d.id);
      if (index == null) return [];

      const predictionInfoEntries: Array<string|number> = [];

      // Does not include prediction info data if it isn't available in
      // keysToTableEntry.
      if (this.keysToTableEntry.size > 0) {
        // Get the classification/regression info cell entries for this row.
        const rowName = d.id;
        // All data entries are passed to the table module, where the columns
        // are filtered before rendering.
        const predictionInfoColumns =
            Array.from(this.columnVisibility.keys())
                .filter((column) => !this.defaultColumns.includes(column));
        predictionInfoColumns.forEach((columnName: string) => {
          const entry =
              this.keysToTableEntry.get(this.getTableKey(rowName, columnName));
          predictionInfoEntries.push((entry == null) ? '-' : entry);
        });
      }

      return [
        index, displayId,
        ...this.keys.map(
            (key) => this.formatForDisplay(d.data[key], this.dataSpec[key])),
        ...predictionInfoEntries
      ];
    });
  }

  firstUpdated() {
    const getCurrentInputData = () => this.appState.currentInputData;
    this.reactImmediately(getCurrentInputData, currentInputData => {
      if (currentInputData != null) {
        this.updatePredictionInfo(currentInputData);
      }
    });
    const getCurrentModels = () => this.appState.currentModels;
    this.react(getCurrentModels, currentModels => {
      this.updateColumns();
    });
    const getCurrentDataset = () => this.appState.currentDataset;
    this.react(getCurrentDataset, currentDataset => {
      this.updateColumns();
    });
    const getKeys = () => this.keys;
    this.react(getKeys, keys => {
      this.updateColumns();
    });

    this.updateColumns();
  }

  private async updatePredictionInfo(currentInputData: IndexedInput[]) {
    const modelPredToClassificationInfo =
        new Map<string, ClassificationInfo[]>();
    const modelPredToRegressionInfo = new Map<string, RegressionInfo[]>();

    const models = this.appState.currentModels;
    const ids = currentInputData.map(data => data.id);

    for (const model of models) {
      const outputSpec = this.appState.currentModelSpecs[model].spec.output;
      const regressionKeys = findSpecKeys(outputSpec, ['RegressionScore']);
      const classificationKeys = findSpecKeys(outputSpec, ['MulticlassPreds']);

      for (const key of classificationKeys) {
        const results =
            await this.classificationService.getResults(ids, model, key);
        modelPredToClassificationInfo.set(
            this.getTableKey(model, key), results);
      }

      for (const key of regressionKeys) {
        const results =
            await this.regressionService.getResults(ids, model, key);
        modelPredToRegressionInfo.set(this.getTableKey(model, key), results);
      }
    }

    this.modelPredToClassificationInfo = modelPredToClassificationInfo;
    this.modelPredToRegressionInfo = modelPredToRegressionInfo;
  }

  private updateColumns() {
    const columnVisibility = new Map<string, boolean>();

    // Add default columns to the map of column names.
    this.defaultColumns.forEach((column) => {
      columnVisibility.set(column, true);
    });

    // Update the map of column names with a possible column for every
    // combination of model, pred key, and classification/regression info type.
    const classificationInfoTypes =
        this.classificationService.getDisplayNames();
    const regressionInfoTypes = this.regressionService.getDisplayNames();

    const models = this.appState.currentModels;
    models.forEach((model) => {
      const outputSpec = this.appState.currentModelSpecs[model].spec.output;
      const regressionKeys = findSpecKeys(outputSpec, ['RegressionScore']);
      this.setColumnNames(
          model, columnVisibility, regressionKeys, regressionInfoTypes);

      const classificationKeys = findSpecKeys(outputSpec, ['MulticlassPreds']);
      this.setColumnNames(
          model, columnVisibility, classificationKeys, classificationInfoTypes);
    });

    this.columnVisibility = columnVisibility;
  }

  /**
   * Adds column to the map of column names for the given model for each
   * combination with pred key and info type.
   */
  private setColumnNames(
      model: string, columnVisibility: Map<string, boolean>, keys: string[],
      infoTypes: string[]) {
    keys.forEach((key) => {
      infoTypes.forEach((type) => {
        columnVisibility.set(this.formatColumnName(model, key, type), false);
      });
    });
  }

  /**
   * Adds table entries for classification/regression info for the given model
   * and keys
   */
  private addTableEntries(
      keysToTableEntry: Map<string, string|number>, model: string,
      keys: string[],
      modelPredToInfo: Map<string, RegressionInfo[]|ClassificationInfo[]>,
      displayNames: string[]) {
    const ids = this.appState.currentInputData.map(data => data.id);
    keys.forEach((key) => {
      const results = modelPredToInfo.get(this.getTableKey(model, key));
      if (results == null) return;

      results.forEach((info: RegressionInfo|ClassificationInfo, i: number) => {
        const entries = Object.entries(info);
        const rowName = ids[i];

        // tslint:disable-next-line:no-any
        entries.forEach((entry: any, i: number) => {
          const displayInfoName = displayNames[i];
          const displayInfoValue = this.formatForDisplay(entry[1]);
          if (displayInfoName == null) return;
          keysToTableEntry.set(
              this.getTableKey(
                  rowName, this.formatColumnName(model, key, displayInfoName)),
              displayInfoValue);
        });
      });
    });
  }

  /**
   * Formats the following types for display in the data table:
   * string, number, boolean, string[], number[], (string|number)[]
   * TODO(lit-dev): allow passing custom HTML to table, not just strings.
   */
  // tslint:disable-next-line:no-any
  formatForDisplay(input: any, fieldSpec?: LitType): string {
    if (input == null) return '';

    // Handle SpanLabels, if field spec given.
    // TODO(lit-dev): handle more fields this way.
    if (fieldSpec != null && isLitSubtype(fieldSpec, 'SpanLabels')) {
      const formattedTags =
          (input as SpanLabel[])
              .map((d: SpanLabel) => `[${d.start}, ${d.end}): ${d.label}`);
      return formattedTags.join(', ');
    }

    // Generic data, based on type of input.
    if (Array.isArray(input)) {
      const maxWordLength = 25;
      const strings = input.map((item) => {
        if (typeof item === 'number') {
          return item.toFixed(4).toString();
        }
        if (typeof item === 'string') {
          return item.length < maxWordLength ?
              item :
              item.substring(0, maxWordLength) + '...';
        }
        return `${item}`;
      });
      return `${strings.join(', ')}`;
    }

    if (typeof input === 'boolean') {
      return input ? '✔' : ' ';
    }

    if (typeof input === 'number') {
      return input.toFixed(4).toString();
    }

    // Fallback: just coerce to string.
    return `${input}`;
  }

  /**
   * Returns the formatted column name for the keyToTableEntry Map.
   */
  private formatColumnName(model: string, key: string, infoType: string) {
    return `${model}:${key}:${infoType}`;
  }

  /**
   * Returns the formatted key for the keyToTableEntry Map from the id and
   * column name.
   */
  private getTableKey(row: string, column: string) {
    return `${row}:${column}`;
  }

  onSelect(selectedRowIndices: number[]) {
    const ids = selectedRowIndices
                    .map(index => this.appState.currentInputData[index]?.id)
                    .filter(id => id != null);
    this.selectionService.selectIds(ids);
  }

  onPrimarySelect(index: number) {
    const id =
        index === -1 ? null : this.appState.currentInputData[index]?.id ?? null;
    this.selectionService.setPrimarySelection(id);
  }

  render() {
    const onSelect = (selectedIndices: number[]) => {
      this.onSelect(selectedIndices);
    };
    const onPrimarySelect = (index: number) => {
      this.onPrimarySelect(index);
    };

    const primarySelectedIndex =
        this.appState.getIndexById(this.selectionService.primarySelectedId);

    // Handle reference selection, if in compare examples mode.
    let referenceSelectedIndex = -1;
    if (this.appState.compareExamplesEnabled) {
      const referenceSelectionService =
          app.getServiceArray(SelectionService)[1];
      referenceSelectedIndex = this.appState.getIndexById(
          referenceSelectionService.primarySelectedId);
    }

    return html`
        <lit-data-table
          .columnVisibility=${this.columnVisibility}
          .data=${this.data}
          .selectedIndices=${this.selectionService.selectedRowIndices}
          .primarySelectedIndex=${primarySelectedIndex}
          .referenceSelectedIndex=${referenceSelectedIndex}
          .onSelect=${onSelect}
          .onPrimarySelect=${onPrimarySelect}
          controlsEnabled
        ></lit-data-table>

    `;
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'data-table-module': DataTableModule;
  }
}
