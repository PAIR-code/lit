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
import '@material/mwc-switch';
import '../elements/checkbox';

import {html} from 'lit';
import {customElement, query} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {DataTable, TableData} from '../elements/table';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {compareArrays, findSpecKeys, shortenId} from '../lib/utils';
import {ClassificationInfo} from '../services/classification_service';
import {RegressionInfo} from '../services/regression_service';
import {ClassificationService, DataService, FocusService, RegressionService, SelectionService} from '../services/services';

import {styles} from './data_table_module.css';

/**
 * A LIT module showing a table containing the InputData examples. Allows the
 * user to sort, filter, and select examples.
 */
@customElement('data-table-module')
export class DataTableModule extends LitModule {
  static override title = 'Data Table';
  static override template = () => {
    return html`<data-table-module></data-table-module>`;
  };
  static override numCols = 4;
  static override get styles() {
    return [sharedStyles, styles];
  }

  static override duplicateForModelComparison = false;

  protected showControls = true;

  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly regressionService = app.getService(RegressionService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);

  @observable columnVisibility = new Map<string, boolean>();
  @observable
  modelPredToClassificationInfo = new Map<string, ClassificationInfo[]>();
  @observable modelPredToRegressionInfo = new Map<string, RegressionInfo[]>();
  @observable searchText = '';

  // Module options / configuration state
  @observable private onlyShowSelected: boolean = false;
  @observable private columnDropdownVisible: boolean = false;

  // Child components
  @query('lit-data-table') private readonly table?: DataTable;

  @computed
  get dataSpec(): Spec {
    return this.appState.currentDatasetSpec;
  }

  @computed
  get keys(): string[] {
    // Use currentInputData to get keys / column names because filteredData
    // might have 0 length;
    const keys = this.appState.currentInputDataKeys;
    const dataKeys = this.dataService.cols.filter(
        col => col.dataType.show_in_data_table).map(col => col.name);
    return keys.concat(dataKeys);
  }

  @computed
  get defaultColumns(): string[] {
    return ['index', ...this.keys];
  }

  @computed
  get filteredData(): IndexedInput[] {
    return this.onlyShowSelected ? this.selectionService.selectedInputData :
                                   this.appState.currentInputData;
  }

  @computed
  get sortedData(): IndexedInput[] {
    // TODO(lit-dev): pre-compute the index chains for each point, since
    // this might get slow if we have a lot of counterfactuals.
    return this.filteredData.slice().sort(
        (a, b) => compareArrays(
            this.reversedAncestorIndices(a), this.reversedAncestorIndices(b)));
  }

  @computed
  get selectedRowIndices(): number[] {
    return this.sortedData
        .map((ex, i) => this.selectionService.isIdSelected(ex.id) ? i : -1)
        .filter(i => i !== -1);
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
  get tableData(): TableData[] {
    const pinnedId = this.appState.compareExamplesEnabled ?
        app.getService(SelectionService, 'pinned').primarySelectedId : null;
    const selectedId = this.selectionService.primarySelectedId;
    const focusedId = this.focusService.focusData?.datapointId;

    // TODO(b/160170742): Make data table render immediately once the
    // non-prediction data is available, then fetch predictions asynchronously
    // and enable the additional columns when ready.
    return this.sortedData.map((d) => {
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
                .filter(
                    (column) => !this.defaultColumns.includes(column) &&
                        this.columnVisibility.get(column));
        predictionInfoColumns.forEach((columnName: string) => {
          const entry =
              this.keysToTableEntry.get(this.getTableKey(rowName, columnName));
          predictionInfoEntries.push((entry == null) ? '-' : entry);
        });
      }

      const dataEntries =
          this.keys.filter(k => this.columnVisibility.get(k))
              .map(k => formatForDisplay(this.dataService.getVal(d.id, k),
                                         this.dataSpec[k]));

      const pinClick = (event: Event) => {
        if (pinnedId === d.id) {
          this.appState.compareExamplesEnabled = false;
        } else {
          this.appState.compareExamplesEnabled = true;
          app.getService(SelectionService, 'pinned').selectIds([d.id]);
        }
        event.stopPropagation();
      };

      const indexHolderDivStyle = styleMap({
        'display': 'flex',
        'flex-direction': 'row-reverse',
        'justify-content': 'space-between',
        'width': '100%'
      });
      const indexDivStyle = styleMap({
        'text-align': 'right',
      });
      // Render the pin button next to the index if datapoint is pinned,
      // selected, or hovered.
      const renderPin = () => {
        const iconClass = classMap({
           'icon-button': true,
           'cyea': true,
           'mdi-outlined': pinnedId !== d.id,
        });
        if (pinnedId === d.id || focusedId === d.id || selectedId === d.id) {
          return html`
              <mwc-icon class="${iconClass}" @click=${pinClick}>
                  push_pin
              </mwc-icon>`;
        }
        return null;
      };
      const indexHtml = html`
          <div style="${indexHolderDivStyle}">
            <div style="${indexDivStyle}">${index}</div>
            ${renderPin()}
          </div>`;
      const indexEntry = {
        template: indexHtml,
        value: index
      };
      const ret: TableData = [indexEntry];
      if (this.columnVisibility.get('id')) {
        ret.push(displayId);
      }
      return [...ret, ...dataEntries, ...predictionInfoEntries];
    });
  }

  override firstUpdated() {
    const getCurrentInputData = () => this.appState.currentInputData;
    this.reactImmediately(getCurrentInputData, currentInputData => {
      if (currentInputData == null) return;
      this.updatePredictionInfo(currentInputData);
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
    columnVisibility.set('id', false);

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
          const displayInfoValue = formatForDisplay(entry[1]);
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

  /**
   * Table callbacks receive indices corresponding to the rows of
   * this.tableData, which matches this.sortedData.
   * We need to map those back to global ids for selection purposes.
   */
  getIdFromTableIndex(tableIndex: number) {
    return this.sortedData[tableIndex]?.id;
  }

  onSelect(tableDataIndices: number[]) {
    const ids = tableDataIndices.map(i => this.getIdFromTableIndex(i))
                    .filter(id => id != null);
    this.selectionService.selectIds(ids, this);
  }

  onPrimarySelect(tableIndex: number) {
    const id = this.getIdFromTableIndex(tableIndex);
    this.selectionService.setPrimarySelection(id, this);
  }

  onHover(tableIndex: number|null) {
    if (tableIndex == null) {
      this.focusService.clearFocus();
    } else {
      const id = this.getIdFromTableIndex(tableIndex);
      this.focusService.setFocusedDatapoint(id);
    }
  }

  renderDropdownItem(key: string) {
    const checked = this.columnVisibility.get(key);
    if (checked == null) return;

    const toggleChecked = () => {
      this.columnVisibility.set(key, !checked);
    };

    // clang-format off
    return html`
      <div>
        <lit-checkbox label=${key} ?checked=${checked}
                      @change=${toggleChecked}>
        </lit-checkbox>
      </div>
    `;
    // clang-format on
  }

  renderColumnDropdown() {
    const names = [...this.columnVisibility.keys()].filter(c => c !== 'index');
    const classes =
        this.columnDropdownVisible ? 'column-dropdown' : 'column-dropdown-hide';
    // clang-format off
    return html`
      <div class='${classes} popup-container'>
        ${names.map(key => this.renderDropdownItem(key))}
      </div>
    `;
    // clang-format on
  }

  renderControls() {
    const onClickResetView = () => {
      this.table!.resetView();
    };

    const onClickSelectAll = () => {
      this.onSelect(this.table!.getVisibleDataIdxs());
    };

    const onToggleShowColumn = () => {
      this.columnDropdownVisible = !this.columnDropdownVisible;
    };

    const onClickSwitch = () => {
      this.onlyShowSelected = !this.onlyShowSelected;
    };

    // clang-format off
    return html`
      <div class='switch-container' @click=${onClickSwitch}>
        <div>Hide unselected</div>
        <mwc-switch .checked=${this.onlyShowSelected}></mwc-switch>
      </div>
      <div id="toolbar-buttons">
        <button class='hairline-button' @click=${onClickResetView}
          ?disabled="${this.table?.isDefaultView ?? true}">
          Reset view
        </button>
        <button class='hairline-button' @click=${onClickSelectAll}>
          Select all
        </button>
        <button class='hairline-button' @click=${onToggleShowColumn}>
          Columns&nbsp;
          <span class='material-icon'>
            ${this.columnDropdownVisible ? "expand_less" : "expand_more"}
          </span>
        </button>
      </div>
      ${this.renderColumnDropdown()}
    `;
    // clang-format on
  }

  renderTable() {
    const tableDataIds = this.sortedData.map(d => d.id);
    const indexOfId = (id: string|null) =>
        id != null ? tableDataIds.indexOf(id) : -1;

    const primarySelectedIndex =
        indexOfId(this.selectionService.primarySelectedId);

    // Set focused index if a datapoint is focused according to the focus
    // service. If the focusData is null then nothing is focused. If focusData
    // contains a value in the "io" field then the focus is on a subfield of
    // a datapoint, as opposed to a datapoint itself.
    const focusData = this.focusService.focusData;
    const focusedIndex = focusData == null || focusData.io != null ?
        -1 :
        indexOfId(focusData.datapointId);

    // Handle reference selection, if in compare examples mode.
    let referenceSelectedIndex = -1;
    if (this.appState.compareExamplesEnabled) {
      const referenceSelectionService =
          app.getService(SelectionService, 'pinned');
      referenceSelectedIndex =
          indexOfId(referenceSelectionService.primarySelectedId);
    }

    const columnNames = [...this.columnVisibility.keys()].filter(
        k => this.columnVisibility.get(k));

    // clang-format off
    return html`
      <lit-data-table
        .data=${this.tableData}
        .columnNames=${columnNames}
        .selectedIndices=${this.selectedRowIndices}
        .primarySelectedIndex=${primarySelectedIndex}
        .referenceSelectedIndex=${referenceSelectedIndex}
        .focusedIndex=${focusedIndex}
        .onSelect=${(idxs: number[]) => { this.onSelect(idxs); }}
        .onPrimarySelect=${(i: number) => { this.onPrimarySelect(i); }}
        .onHover=${(i: number|null)=> { this.onHover(i); }}
        searchEnabled
        selectionEnabled
        paginationEnabled
      ></lit-data-table>
    `;
    // clang-format on
  }

  override render() {
    // clang-format off
    return html`
      <div class='module-container'>
        ${this.showControls ? html`
          <div class='module-toolbar'>
            ${this.renderControls()}
          </div>
        ` : null}
        <div class='module-results-area'>
          ${this.renderTable()}
        </div>
      </div>
    `;
    // clang-format on
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

/**
 * Simplified version of the above; omits toolbar controls.
 */
@customElement('simple-data-table-module')
export class SimpleDataTableModule extends DataTableModule {
  protected override showControls = false;
  static override template = () => {
    return html`<simple-data-table-module></simple-data-table-module>`;
  };
}

declare global {
  interface HTMLElementTagNameMap {
    'data-table-module': DataTableModule;
    'simple-data-table-module': SimpleDataTableModule;
  }
}
