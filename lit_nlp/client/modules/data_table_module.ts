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
import {ColumnHeader, DataTable, TableData} from '../elements/table';
import {LitType, LitTypeWithVocab} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay, IndexedInput, ModelInfoMap, Spec} from '../lib/types';
import {compareArrays, isLitSubtype} from '../lib/utils';
import {DataService, FocusService, SelectionService} from '../services/services';

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

  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);

  @observable columnVisibility = new Map<string, boolean>();
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

  // Column names from the current data for the data table.
  @computed
  get keys(): ColumnHeader[] {
    const createColumnHeader = (name: string, type: LitType) => {
      const header = {name, vocab: (type as LitTypeWithVocab).vocab};
      // TODO(b/162269499): Rename Boolean to BooleanLitType.
      if (isLitSubtype(type, 'Boolean')) {
        header.vocab = ['✔', ' '];
      }
      return header;
    };

    // Use currentInputData to get keys / column names because filteredData
    // might have 0 length;
    const keyNames = this.appState.currentInputDataKeys;
    const keys =
        keyNames.map(key => createColumnHeader(key, this.dataSpec[key]));
    const dataKeys = this.dataService.cols.map(
        col => createColumnHeader(col.name, col.dataType));
    return keys.concat(dataKeys);
  }

  // Filtered keys that hide ones tagged as not to be shown by default in the
  // data table. The filtered ones can still be enabled through the "Columns"
  // selector dropdown.
  @computed
  get defaultKeys(): ColumnHeader[] {
    return this.keys.filter(feat => {
      const col = this.dataService.getColumnInfo(feat.name);
      if (col == null) {
        return true;
      }
      return col.dataType.show_in_data_table;
    });
  }

  // All columns to be available by default in the data table.
  @computed
  get defaultColumns(): ColumnHeader[] {
    return [{name: 'index'}, ...this.keys];
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
  get dataEntries(): Array<Array<string|number>> {
    return this.sortedData.map((d) => {
      const index = this.appState.indicesById.get(d.id);
      if (index == null) return [];

      const dataEntries =
          this.keys.filter(k => this.columnVisibility.get(k.name))
              .map(
                  k => formatForDisplay(
                      this.dataService.getVal(d.id, k.name),
                      this.dataSpec[k.name]));
      return dataEntries;
    });
  }

  @computed
  get selectedRowIndices(): number[] {
    return this.sortedData
        .map((ex, i) => this.selectionService.isIdSelected(ex.id) ? i : -1)
        .filter(i => i !== -1);
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
    return this.dataEntries.map((dataEntry, i) => {
      const d = this.sortedData[i];
      const index = this.appState.indicesById.get(d.id);
      if (index == null) return [];

      const pinClick = (event: Event) => {
        if (pinnedId === d.id) {
          this.appState.compareExamplesEnabled = false;
          app.getService(SelectionService, 'pinned').selectIds([]);
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
      return [...ret, ...dataEntry];
    });
  }

  override firstUpdated() {
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

  private updateColumns() {
    const columnVisibility = new Map<string, boolean>();

    // Add default columns to the map of column names.
    for (const column of this.defaultColumns) {
      columnVisibility.set(
          column.name,
          this.defaultKeys.includes(column) || column.name === 'index');
    }
    this.columnVisibility = columnVisibility;
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

    const columnNames =
        this.defaultColumns.filter(col => this.columnVisibility.get(col.name));

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
