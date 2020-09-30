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

/**
 * A generic Data Table component that can be reused across the
 * Data Table / Metrics Modules
 */

// tslint:disable:no-new-decorators
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import '@material/mwc-icon';

import {ascending, descending} from 'd3';  // array helpers.
import {customElement, html, property} from 'lit-element';
import {TemplateResult} from 'lit-html';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';
import {ReactiveElement} from '../lib/elements';

import {styles} from './table.css';

/** Wrapper types for the data supplied to the data table */
export type TableEntry = string|number|TemplateResult;
/** Wrapper types for the data supplied to the data table */
export type TableData = TableEntry[];

/** Callback for selection */
export type OnSelectCallback = (selectedIndices: number[]) => void;
/** Callback for primary datapoint selection */
export type OnPrimarySelectCallback = (index: number) => void;
/** Function for reading data index from a row */
export type GetDataIndexFromRowFunction = (row: TableData) => number;
/** Function for sorting a particular column by a value that's
 * different than what's displayed */
export type GetSortValue = (d: TableData, columnIndex: number) => any;


enum SpanAnchor {
  START,
  END,
}


/**
 * A generic data table component
 */
@customElement('lit-data-table')
export class DataTable extends ReactiveElement {
  @observable @property({type: Array}) data: TableData[] = [];
  @property({type: Array}) selectedIndices: number[] = [];
  @property({type: Number}) primarySelectedIndex: number = -1;
  @property({type: Number}) referenceSelectedIndex: number = -1;
  @property({type: Boolean}) selectionDisabled: boolean = false;
  @property({type: Boolean}) controlsEnabled: boolean = false;
  @observable
  @property({type: Object})
  columnVisibility = new Map<string, boolean>();
  @property({type: String}) defaultSortName = 'default';
  @property({type: Boolean}) defaultSortAscending = true;

  // Callbacks
  @property({type: Object}) onSelect: OnSelectCallback = () => {};
  @property({type: Object}) onPrimarySelect: OnPrimarySelectCallback = () => {};
  @property({type: Object}) getDataIndexFromRow: GetDataIndexFromRowFunction = d => +d[0]
  @property({type: Object}) getSortValue: GetSortValue = (d, columnIndex) => d[columnIndex]

  static get styles() {
    return [styles];
  }

  // If sortName is undefined, we use the order of the input data.
  @observable private sortName?: string;
  @observable private sortAscending = true;
  @observable private showColumnMenu = false;
  @observable private columnMenuName = '';
  @observable private readonly columnSearchQueries = new Map<string, string>();
  @observable private filterSelected = false;
  @observable columnDropdownVisible = false;
  @observable private headerWidths: number[] = [];

  // Sorted data. We manage updates with a reaction to enable "sticky" behavior.
  private stickySortedData?: TableData[]|null = null;

  private resizeObserver!: ResizeObserver;

  private selectedIndicesSetForRender = new Set<number>();
  private rowIndexToDataIndex = new Map<number, number>();

  private shiftSelectionStartIndex = 0;
  private shiftSelectionEndIndex = 0;
  private shiftSpanAnchor = SpanAnchor.START;

  firstUpdated() {
    const container = this.shadowRoot!.getElementById('rows')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
    this.resizeObserver.observe(container);

    // Clear "sticky" sorted data if the inputs change.
    this.reactImmediately(() => this.rowFilteredData, filteredData => {
      this.stickySortedData = null;
    });

    // TODO(lit-dev) not sure why this isn't working on its own
    this.sortName = this.defaultSortName;
    this.sortAscending = this.defaultSortAscending;
    this.requestUpdate();
  }

  // tslint:disable-next-line:no-any
  shouldUpdate(changedProperties: any) {
    if (changedProperties.get('data')) {
      // Let's just punt on the issue of maintaining shift selection behavior
      // when the data changes (via filtering, for example)
      this.shiftSelectionStartIndex = 0;
      this.shiftSelectionEndIndex = 0;
    }
    return true;
  }

  private resize() {
    // Compute the table header sizes based on the table layout
    // tslint:disable-next-line:no-any (can't iterate over HTMLCollection...)
    const row: any = this.shadowRoot!.querySelector('tr');
    if (row) {
      this.headerWidths = [...row.children].map((child: HTMLElement) => {
        return child.getBoundingClientRect().width;
      });
    }
  }

  @computed
  get columnNames(): string[] {
    return Array.from(this.columnVisibility.keys());
  }

  @computed
  get sortIndex(): number {
    const sortName = this.sortName || this.defaultSortName;
    return (sortName == null) ? 0 : this.columnNames.indexOf(sortName);
  }

  /**
   * This computed returns the data filtered by row (filtering by column
   * happens in render()).
   */
  @computed
  get rowFilteredData(): TableData[] {
    const data = this.data.slice();
    const selectedIndices = new Set<number>(this.selectedIndices);

    const rowFilteredData = data.filter((item) => {
      let isShownByTextFilter = true;
      // Apply column search filters
      for (const [key, value] of this.columnSearchQueries) {
        const index = this.columnNames.indexOf(key);
        if (index === -1) return;

        const col = item[index];
        if (typeof col === 'string') {
          // TODO(b/158299036) Change this to regexp search.
          isShownByTextFilter = isShownByTextFilter && col.includes(value);
        } else if (typeof col === 'number') {
          // TODO(b/158299036) Support syntax like 1-3,6 for numbers.
          isShownByTextFilter = isShownByTextFilter && value === '' ?
              true :
              col.toString() === value;
        }
      }

      let isShownBySelectedFilter = true;
      if (this.filterSelected) {
        const areSomeSelected = this.selectedIndices.length > 0;
        isShownBySelectedFilter =
            areSomeSelected ? selectedIndices.has(+item[0]) : true;
      }
      return isShownByTextFilter && isShownBySelectedFilter;
    });
    return rowFilteredData;
  }

  getSortedData(): TableData[] {
    const source = this.stickySortedData ?? this.rowFilteredData;
    let sortedData = source.slice();
    sortedData = sortedData.sort((a, b) => {
      const compareFn = (this.sortAscending ? ascending : descending);
      const aValue = this.getSortValue(a, this.sortIndex);
      const bValue = this.getSortValue(b, this.sortIndex);
      return compareFn(aValue, bValue);
    });

    // Store a mapping from the row to data indices.
    this.rowIndexToDataIndex =
        new Map(sortedData.map((d, index) => [index, this.getDataIndexFromRow(d)]));

    this.stickySortedData = sortedData;
    return sortedData;
  }

  private setShiftSelectionSpan(startIndex: number, endIndex: number) {
    this.shiftSelectionStartIndex = startIndex;
    this.shiftSelectionEndIndex = endIndex;
  }

  private isRowWithinShiftSelectionSpan(index: number) {
    return index >= this.shiftSelectionStartIndex &&
        index <= this.shiftSelectionEndIndex;
  }

  private selectFromRange(
      selectedIndices: Set<number>, start: number, end: number, select = true) {
    for (let rowIndex = start; rowIndex <= end; rowIndex++) {
      const dataIndex = this.rowIndexToDataIndex.get(rowIndex);
      if (dataIndex == null) return;

      if (select) {
        selectedIndices.add(dataIndex);
      } else {
        selectedIndices.delete(dataIndex);
        // If removing the primary selected point from a selection,
        // reset the primary selected index.
        if (dataIndex === this.primarySelectedIndex) {
          this.setPrimarySelectedIndex(-1);
        }
      }
    }
  }

  /** Logic for handling row / multirow selection */
  private handleRowClick(e: MouseEvent, rowIndex: number) {
    let selectedIndices = new Set<number>(this.selectedIndices);
    let doChangeSelectedSet = true;

    const dataIndex = this.rowIndexToDataIndex.get(rowIndex);
    if (dataIndex == null) return;

    // Handle ctrl/cmd-click
    if (e.metaKey || e.ctrlKey) {
      if (selectedIndices.has(dataIndex)) {
        selectedIndices.delete(dataIndex);
        // If removing the primary selected point from a selection,
        // reset the primary selected index.
        if (dataIndex === this.primarySelectedIndex) {
          this.setPrimarySelectedIndex(-1);
        }
      } else {
        selectedIndices.add(dataIndex);
      }
      this.setShiftSelectionSpan(rowIndex, rowIndex);
    }
    //  Rather complicated logic for handling shift-click
    else if (e.shiftKey) {
      // Prevent selecting text on shift-click
      e.preventDefault();

      const isWithinSpan = this.isRowWithinShiftSelectionSpan(rowIndex);
      const startIndex = this.shiftSelectionStartIndex;
      const endIndex = this.shiftSelectionEndIndex;

      if (isWithinSpan) {
        if (this.shiftSpanAnchor === SpanAnchor.START) {
          this.selectFromRange(selectedIndices, rowIndex + 1, endIndex, false);
          this.shiftSelectionEndIndex = rowIndex;
        } else {
          this.selectFromRange(
              selectedIndices, startIndex, rowIndex - 1, false);
          this.shiftSelectionStartIndex = rowIndex;
        }
      } else {
        if (rowIndex >= endIndex) {
          this.selectFromRange(selectedIndices, endIndex, rowIndex);
          this.setShiftSelectionSpan(startIndex, rowIndex);
          this.shiftSpanAnchor = SpanAnchor.START;
        } else if (rowIndex <= startIndex) {
          this.selectFromRange(selectedIndices, rowIndex, startIndex);
          this.setShiftSelectionSpan(rowIndex, endIndex);
          this.shiftSpanAnchor = SpanAnchor.END;
        }
      }
    }
    // Otherwise, simply select/deselect the index.
    else {
      if (selectedIndices.has(dataIndex) && selectedIndices.size === 1) {
        selectedIndices = new Set<number>();
        this.setPrimarySelectedIndex(-1);
      } else if (
          selectedIndices.has(dataIndex) &&
          dataIndex !== this.primarySelectedIndex) {
        // If selecting a different primary selected index inside of
        // a previously-created selection, then change the primary selection
        // but do not update the overall list of selected points.
        this.setPrimarySelectedIndex(rowIndex);
        doChangeSelectedSet = false;
      } else {
        selectedIndices = new Set<number>([dataIndex]);
        this.setPrimarySelectedIndex(rowIndex);
      }
      this.setShiftSelectionSpan(rowIndex, rowIndex);
    }
    if (doChangeSelectedSet) {
      this.selectedIndices = Array.from(selectedIndices);
      this.onSelect([...this.selectedIndices]);
    }
  }

  private setPrimarySelectedIndex(rowIndex: number) {
    let primaryIndex = -1;
    if (rowIndex !== -1) {
      const dataIndex = this.rowIndexToDataIndex.get(rowIndex);
      if (dataIndex == null) return;

      primaryIndex = dataIndex;
    }
    this.primarySelectedIndex = primaryIndex;
    this.onPrimarySelect(primaryIndex);
  }

  render() {
    // Make a private, temporary set of selectedIndices to simplify lookup
    // in the row render method
    this.selectedIndicesSetForRender = new Set<number>(this.selectedIndices);

    const data = this.getSortedData();
    const columns = Array.from(this.columnVisibility.keys());

    // Only show columns that are set as visible in the column dropdown.
    const columnFilteredData = data.map((row) => {
      return row.filter((entry, i) => {
        return this.columnVisibility.get(columns[i]);
      });
    });

    // Synchronizes the horizontal scrolling of the header with the rows.
    const onScroll = (e: Event) => {
      const header = this.shadowRoot!.getElementById('header-container');
      const body = e.target as HTMLElement;
      if (header != null && body != null) {
        header.scrollLeft = body.scrollLeft;
      }
    };

    const onClickSelectAll = () => {
      this.selectedIndices = columnFilteredData.map((d, index) => +d[0]);
      this.onSelect([...this.selectedIndices]);
    };

    const isDefaultView = (
      (this.sortName === this.defaultSortName) &&
      (this.sortAscending === this.defaultSortAscending) &&
      (this.columnSearchQueries.size === 0) &&
      !this.filterSelected
    );
    const onClickResetView = () => {
      this.columnSearchQueries.clear();
      this.sortName = this.defaultSortName;
      this.sortAscending = this.defaultSortAscending;
      this.filterSelected = false;
    };

    const toggleFilterSelected = () => {
      this.filterSelected = !this.filterSelected;
    };

    const onToggleShowColumn = () => {
      this.columnDropdownVisible = !this.columnDropdownVisible;
    };

    const visibleColumns =
        this.columnNames.filter((key) => this.columnVisibility.get(key));

    // clang-format off
    return html`
      <div id="holder">
        ${this.controlsEnabled ? html`
        <div class="toolbar">
          <lit-checkbox
            label="Only show selected"
            ?checked=${this.filterSelected}
            @change=${toggleFilterSelected}
          ></lit-checkbox>
          <div id="toolbar-buttons">
            <button id="default-view" @click=${onClickResetView}
              ?disabled="${isDefaultView}">
              Reset view
            </button>
            <button id="select-all" @click=${onClickSelectAll}>
              Select all
            </button>
            <button id="column-button" @click=${onToggleShowColumn}>
              Columns
              <span data-icon=${this.columnDropdownVisible ? "expand_less" :
              "expand_more"}></span>
            </button>
          </div>
          ${this.renderColumnDropdown()}
        </div>` : null}
        <div id="table-container">
          <div id="header-container">
            <div id="header">
              ${visibleColumns.map((c, i) => this.renderColumnHeader(c, i))}
            </div>
          </div>
          <div id="rows-container" @scroll=${onScroll}>
            <table id="rows">
              <tbody>
                ${columnFilteredData.map((d, rowIndex) => this.renderRow(d, rowIndex))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  renderColumnHeader(title: string, index: number) {
    // this.headerWidths sometimes hasn't been updated when this method is
    // called since it's set in this.resize() which uses the table cells'
    // clientWidth to set this.headerWidths. Return if the index is out of
    // bounds.
    if (index >= this.headerWidths.length) return;
    const headerWidth = this.headerWidths[index];
    const width = headerWidth ? `${headerWidth}px` : '';
    let searchText = this.columnSearchQueries.get(title);
    if (searchText === undefined) {
      searchText = '';
    }

    const handleClick = () => {
      if (this.sortName === title) {
        this.sortAscending = !this.sortAscending;
      } else {
        this.sortName = title;
        this.sortAscending = true;
      }
    };

    const isSearchActive = () => {
      const searchString = this.columnSearchQueries.get(title);
      if (searchString ||
          (this.showColumnMenu && this.columnMenuName === title)) {
        return true;
      }
      return false;
    };

    const searchMenuStyle = styleMap({
      width,
      'visibility':
          (this.showColumnMenu && this.columnMenuName === title ? 'visible' :
                                                                  'hidden'),
    });

    const menuButtonStyle =
        styleMap({'outline': (isSearchActive() ? 'auto' : 'none')});

    const handleMenuButton = () => {
      if (this.columnMenuName === title) {
        this.showColumnMenu = !this.showColumnMenu;
      } else {
        this.columnMenuName = title;
        this.showColumnMenu = true;
      }
    };

    const handleSearchChange = (e: KeyboardEvent) => {
      this.columnSearchQueries.set(
          title, (e.target as HTMLInputElement)?.value || '');
    };

    const isUpActive = this.sortName === title && this.sortAscending;
    const isUpInactive = this.sortName === title && !this.sortAscending;
    const isDownActive = this.sortName === title && !this.sortAscending;
    const isDownInactive = this.sortName === title && this.sortAscending;

    const upArrowClasses = classMap({
      arrow: true,
      up: true,
      active: isUpActive,
      inactive: isUpInactive,
    });
    const downArrowClasses = classMap({
      arrow: true,
      down: true,
      active: isDownActive,
      inactive: isDownInactive,
    });

    const style = styleMap({width});
    // clang-format off
    return html`
        <div>
          <div class="column-header" title=${title} style=${style}>
            <div class="header-text">${title}</div>
            ${this.controlsEnabled ? html`
            <div class="menu-button-container">
              <mwc-icon class="menu-button" style=${menuButtonStyle} @click=${handleMenuButton}>search</mwc-icon>
            </div>` : null}
            <div class="arrow-container" @click=${handleClick}>
              <mwc-icon class=${upArrowClasses}>arrow_drop_up</mwc-icon>
              <mwc-icon class=${downArrowClasses}>arrow_drop_down</mwc-icon>
            </div>
          </div>
          ${this.controlsEnabled ? html`
          <div class='togglable-menu-holder' style=${searchMenuStyle}>
              <input type="search" id='search-menu-container'
              .value=${searchText}
              placeholder="Search" @input=${handleSearchChange}/>
          </div>` : null}
        </div>
      `;
    // clang-format on
  }


  renderRow(data: TableData, rowIndex: number) {
    const dataIndex = this.rowIndexToDataIndex.get(rowIndex);
    if (dataIndex == null) return;

    const isSelected = this.selectedIndicesSetForRender.has(dataIndex);
    const isPrimarySelection = this.primarySelectedIndex === dataIndex;
    const isReferenceSelection = this.referenceSelectedIndex === dataIndex;
    const rowClass = classMap({
      'selected': isSelected,
      'primary-selected': isPrimarySelection,
      'reference-selected': isReferenceSelection,
    });
    const mouseDown = (e: MouseEvent) => {
      if (this.selectionDisabled) return;
      this.handleRowClick(e, rowIndex);
    };
    return html`
        <tr class="${rowClass}" @mousedown=${mouseDown}>
          ${data.map(d => html`<td>${d}</td>`)}
        </tr>
      `;
  }

  renderColumnDropdown() {
    // clang-format off
    return html`
        <div class='${this.columnDropdownVisible ? 'column-dropdown' :
          'column-dropdown-hide'}'>
          ${this.columnNames.filter((column) => column !== 'index')
            .map(key => this.renderDropdownItem(key))}
        </div>
    `;
    // clang-format on
  }

  renderDropdownItem(key: string) {
    const checked = this.columnVisibility.get(key);
    if (checked == null) return;

    const toggleChecked = () => {
      this.columnVisibility.set(key, !checked);

      const resize = () => {
        this.resize();
      };
      // Delays the call that resizes the column widths, since the resize
      // depends on the data table cell widths which might get updated by the
      // first render call.
      requestAnimationFrame(resize);
    };

    return html`
        <div>
          <lit-checkbox
              label=${key}
              ?checked=${checked}
              @change=${toggleChecked}
            >
          </lit-checkbox>
        </div>
        `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-data-table': DataTable;
  }
}
