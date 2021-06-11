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
import {customElement, html, property, TemplateResult} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {isTemplateResult} from 'lit-html/directive-helpers';
import {action, computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {formatForDisplay} from '../lib/types';
import {isNumber} from '../lib/utils';

import {styles} from './table.css';

type SortableTableEntry = string|number;
/** Wrapper type for sortable custom data table entries */
export interface SortableTemplateResult {
  template: TemplateResult;
  value: SortableTableEntry;
}
/** Wrapper types for the data supplied to the data table */
export type TableEntry = string|number|TemplateResult|SortableTemplateResult;
/** Wrapper types for the data supplied to the data table */
export type TableData = TableEntry[]|{[key: string]: TableEntry};


/** Internal data, including metadata */
interface TableRowInternal {
  inputIndex: number; /* index in original this.data */
  rowData: TableEntry[];
}

/** Callback for selection */
export type OnSelectCallback = (selectedIndices: number[]) => void;
/** Callback for primary datapoint selection */
export type OnPrimarySelectCallback = (index: number) => void;
/** Callback for hover */
export type OnHoverCallback = (index: number|null) => void;

enum SpanAnchor {
  START,
  END,
}

const IMAGE_PREFIX = 'data:image';

const CONTAINER_HEIGHT_CHANGE_DELTA = 20;

/**
 * A generic data table component
 */
@customElement('lit-data-table')
export class DataTable extends ReactiveElement {
  // observable.struct is necessary to avoid spurious updates
  // if this object identity changes. This can happen if plain JS data is
  // passed to this property, as it will subsequently be proxied by mobx.
  // The structural comparison ensures that this proxying will not trigger
  // updates if the underlying data does not change.
  // TODO(lit-dev): investigate any performance implications of this deep
  // comparison, as this may run more frequently than we'd like.
  // TODO(lit-dev): consider observable.ref or observable.shallow;
  // see https://mobx.js.org/observable-state.html#available-annotations.
  // This could save performance, since calling code can always do [...data]
  // to generate a new reference and force a refresh if needed.
  @observable.struct @property({type: Array}) data: TableData[] = [];
  @observable.struct @property({type: Array}) columnNames: string[] = [];
  @observable.struct @property({type: Array}) selectedIndices: number[] = [];
  @observable @property({type: Number}) primarySelectedIndex: number = -1;
  @observable @property({type: Number}) referenceSelectedIndex: number = -1;
  // TODO(lit-dev): consider a custom reaction to make this more responsive,
  // instead of triggering a full re-render.
  @observable @property({type: Number}) focusedIndex: number = -1;

  // Mode controls
  @observable @property({type: Boolean}) selectionEnabled: boolean = false;
  @observable @property({type: Boolean}) searchEnabled: boolean = false;
  @observable @property({type: Boolean}) paginationEnabled: boolean = false;

  // Callbacks
  @property({type: Object}) onClick: OnPrimarySelectCallback|undefined;
  @property({type: Object}) onHover: OnHoverCallback|undefined;
  @property({type: Object}) onSelect: OnSelectCallback = () => {};
  @property({type: Object}) onPrimarySelect: OnPrimarySelectCallback = () => {};

  static get styles() {
    return [styles];
  }

  // If sortName is undefined, we use the order of the input data.
  @observable private sortName?: string;
  @observable private sortAscending = true;
  @observable private showColumnMenu = false;
  @observable private columnMenuName = '';
  @observable private readonly columnSearchQueries = new Map<string, string>();
  @observable private headerWidths: number[] = [];
  @observable private pageNum = 0;
  @observable private entriesPerPage = 10;

  // Sorted data. We manage updates with a reaction to enable "sticky" behavior,
  // where subsequent sorts are based on the last sort rather than the original
  // inputs (i.e. this.data). This way, you can do useful compound sorts like in
  // a typical spreadsheet program.
  @observable private stickySortedData?: TableRowInternal[]|null = null;

  private resizeObserver!: ResizeObserver;
  private needsEntriesPerPageRecompute = true;
  private lastContainerHeight = 0;

  private selectedIndicesSetForRender = new Set<number>();

  private shiftSelectionStartIndex = 0;
  private shiftSelectionEndIndex = 0;
  private shiftSpanAnchor = SpanAnchor.START;
  private hoveredIndex: number|null = null;

  firstUpdated() {
    const container = this.shadowRoot!.querySelector('.rows-container')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.computeHeaderWidths();
      this.adjustEntriesIfHeightChanged();
    });
    this.resizeObserver.observe(container);

    // If inputs changed, re-sort data based on the new inputs.
    this.reactImmediately(() => this.rowFilteredData, filteredData => {
      this.stickySortedData = null;
      this.needsEntriesPerPageRecompute = true;
      this.pageNum = 0;
      this.requestUpdate();
    });
    // If sort settings are changed, re-sort data optionally using result of
    // previous sort.
    const triggerSort = () => [this.sortName, this.sortAscending];
    this.reactImmediately(triggerSort, () => {
      this.stickySortedData = this.getSortedData(this.displayData);
      this.pageNum = 0;
      this.requestUpdate();
    });
    // Reset page number if invalid on change in total pages.
    const triggerPageChange = () => this.totalPages;
    this.reactImmediately(triggerPageChange, () => {
      if (this.pageNum >= this.totalPages) {
        this.pageNum = 0;
      }
    });
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

  updated() {
    if (this.needsEntriesPerPageRecompute) {
      this.computeEntriesPerPage();
    }
  }

  private getRowsContainerHeight() {
    const container: HTMLElement =
        // tslint:disable-next-line:no-unnecessary-type-assertion
        this.shadowRoot!.querySelector('.rows-container')! as HTMLElement;
    return container.getBoundingClientRect().height;
  }

  // If the row container's height has changed significantly, then recompute
  // entries per row.
  private adjustEntriesIfHeightChanged() {
    const containerHeight = this.getRowsContainerHeight();
    if (Math.abs(containerHeight - this.lastContainerHeight) >
        CONTAINER_HEIGHT_CHANGE_DELTA) {
      this.lastContainerHeight = containerHeight;
      this.computeEntriesPerPage();
    }
  }

  private computeEntriesPerPage() {
    this.needsEntriesPerPageRecompute = false;

    // If pagination is disabled, then ensure all entires can fit on a single
    // page.
    if (!this.paginationEnabled) {
      this.entriesPerPage = this.rowFilteredData.length + 1;
      return;
    }

    const containerHeight = this.getRowsContainerHeight();
    const rows: NodeListOf<HTMLElement> =
        this.shadowRoot!.querySelectorAll('tr');
    let height = 0;
    let i = 0;

    // Iterate over rows, adding up their height until they fill the container,
    // to get the number of rows to display per page.
    for (i = 0; i < rows.length; i++) {
      height += rows[i].getBoundingClientRect().height;
      if (height > containerHeight) {
        this.entriesPerPage = i + 1;
        break;
      }
    }
    if (height === 0) {
      this.entriesPerPage = 10;
    } else if (height <= containerHeight) {
      // If there aren't enough entries to take up the entire container,
      // calculate how many will fill the container based on the heights so far.
      const heightPerEntry = height / i;
      this.entriesPerPage = Math.ceil(containerHeight / heightPerEntry);
    }
    // Round up to the nearest 10.
    this.entriesPerPage = Math.ceil(this.entriesPerPage / 10) * 10;
  }

  private computeHeaderWidths() {
    // Compute the table header sizes based on the table layout
    // tslint:disable-next-line:no-any (can't iterate over HTMLCollection...)
    const row: any = this.shadowRoot!.querySelector('tr');
    if (row) {
      this.headerWidths = [...row.children].map((child: HTMLElement) => {
        return child.getBoundingClientRect().width;
      });
    }
  }

  private getSortableEntry(colEntry: TableEntry): SortableTableEntry {
    // Passthrough values if TableEntry is number or string. If it is
    // TemplateResult return 0 for sorting purposes. If it is a sortable
    // tempate result then sort by the underlying sortable value.
    if (typeof colEntry === "string" || isNumber(colEntry)) {
      return colEntry as SortableTableEntry;
    }
    if (isTemplateResult(colEntry)) {
      return 0;
    }
    return (colEntry as SortableTemplateResult).value;
  }

  @computed
  get sortIndex(): number|undefined {
    return (this.sortName == null) ? undefined :
                                     this.columnNames.indexOf(this.sortName);
  }

  /**
   * First pass processing input data to canonical form.
   * The data goes through several stages before rendering:
   * - this.data is the input data (bound via element property)
   * - indexedData converts to parallel-list form and adds input indices
   *   for later reference
   * - rowFilteredData filters to a subset of rows, based on search criteria
   * - getSortedData() is called in a reaction to sort this if sort-by-column
   *   is used. The result is stored in this.stickySortedData so that future
   *   sorts can be "stable" rather than re-sorting from the input.
   * - displayData is stickySortedData, or rowFilteredData if that is unset.
   *   This is used to actually render the table.
   */
  @computed
  get indexedData(): TableRowInternal[] {
    // Convert any objects to simple arrays by selecting fields.
    const convertedData: TableEntry[][] = this.data.map((d: TableData) => {
      if (d instanceof Array) return d;
      return this.columnNames.map(k => d[k]);
    });
    return convertedData.map((rowData: TableEntry[], inputIndex: number) => {
      return {inputIndex, rowData};
    });
  }

  /**
   * This computed returns the data filtered by row.
   */
  @computed
  get rowFilteredData(): TableRowInternal[] {
    return this.indexedData.filter((item) => {
      let isShownByTextFilter = true;
      // Apply column search filters
      for (const [key, value] of this.columnSearchQueries) {
        const index = this.columnNames.indexOf(key);
        if (index === -1) return;

        const col = item.rowData[index];
        if (typeof col === 'string') {
          isShownByTextFilter =
              isShownByTextFilter && col.search(new RegExp(value)) !== -1;
        } else if (typeof col === 'number') {
          // TODO(b/158299036) Support syntax like 1-3,6 for numbers.
          isShownByTextFilter = isShownByTextFilter && value === '' ?
              true :
              col.toString() === value;
        }
      }
      return isShownByTextFilter;
    });
  }

  getSortedData(source: TableRowInternal[]): TableRowInternal[] {
    let sortedData = source.slice();
    if (this.sortName != null) {
      sortedData = sortedData.sort(
          (a, b) => (this.sortAscending ? ascending : descending)(
              this.getSortableEntry(a.rowData[this.sortIndex!]),
              this.getSortableEntry(b.rowData[this.sortIndex!])));
    }
    return sortedData;
  }

  @computed
  get displayData(): TableRowInternal[] {
    return this.stickySortedData ?? this.rowFilteredData;
  }

  @computed
  get totalPages(): number {
    return Math.ceil(this.displayData.length / this.entriesPerPage);
  }

  // The entries that fit on the current page in the table.
  @computed
  get pageData(): TableRowInternal[] {
    const begin = this.pageNum * this.entriesPerPage;
    const end = begin + this.entriesPerPage;
    return this.displayData.slice(begin, end);
  }

  @computed
  get hasFooter(): boolean {
    return this.displayData.length > this.entriesPerPage;
  }

  private setShiftSelectionSpan(startIndex: number, endIndex: number) {
    this.shiftSelectionStartIndex = startIndex;
    this.shiftSelectionEndIndex = endIndex;
  }

  private isRowWithinShiftSelectionSpan(index: number) {
    return index >= this.shiftSelectionStartIndex &&
        index <= this.shiftSelectionEndIndex;
  }

  private getInputIndexFromRowIndex(rowIndex: number) {
    return this.displayData[rowIndex].inputIndex;
  }

  private selectFromRange(
      selectedIndices: Set<number>, start: number, end: number, select = true) {
    for (let rowIndex = start; rowIndex <= end; rowIndex++) {
      const dataIndex = this.getInputIndexFromRowIndex(rowIndex);
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
  @action
  private handleRowClick(e: MouseEvent, dataIndex: number, rowIndex: number) {
    let selectedIndices = new Set<number>(this.selectedIndices);
    let doChangeSelectedSet = true;

    if (this.onClick != null) {
      this.onClick(dataIndex);
      return;
    }

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

  /** Logic for handling row hover */
  private handleRowMouseEnter(e: MouseEvent, dataIndex: number) {
    this.hoveredIndex = dataIndex;
    if (this.onHover != null) {
      this.onHover(this.hoveredIndex);
      return;
    }
  }
  private handleRowMouseLeave(e: MouseEvent, dataIndex: number) {
    if (dataIndex === this.hoveredIndex) {
      this.hoveredIndex = null;
    }

    if (this.onHover != null) {
      this.onHover(this.hoveredIndex);
      return;
    }
  }

  @action
  private setPrimarySelectedIndex(rowIndex: number) {
    let primaryIndex = -1;
    if (rowIndex !== -1) {
      const dataIndex = this.getInputIndexFromRowIndex(rowIndex);
      if (dataIndex == null) return;

      primaryIndex = dataIndex;
    }
    this.primarySelectedIndex = primaryIndex;
    this.onPrimarySelect(primaryIndex);
  }

  /**
   * Imperative controls, intended to be used by a containing module
   * such as data_table_module.ts
   */
  @computed
  get isDefaultView() {
    return this.sortName === undefined && this.columnSearchQueries.size === 0;
  }

  resetView() {
    this.columnSearchQueries.clear();
    this.sortName = undefined;  // reset to input ordering
    this.showColumnMenu = false;  // hide search bar
    // Reset sticky sort and re-render from input data.
    this.stickySortedData = null;
    this.requestUpdate();
  }

  getVisibleDataIdxs(): number[] {
    return this.displayData.map(d => d.inputIndex);
  }

  render() {
    // Make a private, temporary set of selectedIndices to simplify lookup
    // in the row render method
    this.selectedIndicesSetForRender = new Set<number>(this.selectedIndices);

    // Synchronizes the horizontal scrolling of the header with the rows.
    const onScroll = (e: Event) => {
      const header = this.shadowRoot!.getElementById('header-container');
      const body = e.target as HTMLElement;
      if (header != null && body != null) {
        header.scrollLeft = body.scrollLeft;
      }
    };

    //TODO: calculate correct # of entries per page.
    const rowsContainerClasses = classMap({
      'rows-container': true,
      'with-footer': this.hasFooter
    });
    // clang-format off
    return html`
      <div id="holder">
        <div id="header-container">
          <div id="header">
            ${this.columnNames.map((c, i) => this.renderColumnHeader(c, i))}
          </div>
        </div>
        <div class=${rowsContainerClasses} @scroll=${onScroll}>
          <table id="rows">
            <tbody>
              ${this.pageData.map((d, rowIndex) => this.renderRow(d, rowIndex))}
            </tbody>
          </table>
        </div>
        ${this.renderFooter()}
      </div>
    `;
    // clang-format on
  }

  renderFooter() {
    if (!this.hasFooter) {
      return null;
    }
    const pageDisplayNum = this.pageNum + 1;
    // Use this modulo function so that if pageNum is negative (when
    // decrementing pages), the modulo returns the expected positive value.
    const modPageNumber = (pageNum: number) => {
      return ((pageNum % this.totalPages) + this.totalPages) % this.totalPages;
    };
    const nextPage = () => {
      const newPageNum = modPageNumber(this.pageNum + 1);
      this.pageNum = newPageNum;
    };
    const prevPage = () => {
      const newPageNum = modPageNumber(this.pageNum -1);
      this.pageNum = newPageNum;
    };
    const firstPage = () => {
      this.pageNum = 0;
    };
    const firstPageButtonClasses = {
      'icon-button': true,
      'disabled': this.pageNum === 0
    };
    // clang-format off
    return html`
      <div id="footer-container">
        <div id="footer">
          <mwc-icon class=${classMap(firstPageButtonClasses)}
            @click=${firstPage}>
            first_page
          </mwc-icon>
          <mwc-icon class='icon-button'
            @click=${prevPage}>
            chevron_left
          </mwc-icon>
          <div>
           Page
           <span class="current-page-num">${pageDisplayNum}</span>
           of ${this.totalPages}
          </div>
          <mwc-icon class='icon-button'
             @click=${nextPage}>
            chevron_right
          </mwc-icon>
        </div>
      </div>`;
  }

  renderColumnHeader(title: string, index: number) {
    // this.headerWidths sometimes hasn't been updated when this method is
    // called since it's set in this.computeHeaderWidths() which uses the
    // table cells' clientWidth to set this.headerWidths.
    // Return if the index is out of bounds.
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
            ${this.searchEnabled ? html`
            <div class="menu-button-container">
              <mwc-icon class="menu-button" style=${menuButtonStyle} @click=${handleMenuButton}>search</mwc-icon>
            </div>` : null}
            <div class="arrow-container" @click=${handleClick}>
              <mwc-icon class=${upArrowClasses}>arrow_drop_up</mwc-icon>
              <mwc-icon class=${downArrowClasses}>arrow_drop_down</mwc-icon>
            </div>
          </div>
          ${this.searchEnabled ? html`
          <div class='togglable-menu-holder' style=${searchMenuStyle}>
              <input type="search" id='search-menu-container'
              .value=${searchText}
              placeholder="Search" @input=${handleSearchChange}/>
          </div>` : null}
        </div>
      `;
    // clang-format on
  }


  renderRow(data: TableRowInternal, rowIndex: number) {
    const dataIndex = data.inputIndex;
    const displayDataIndex = rowIndex + this.pageNum * this.entriesPerPage;

    const isSelected = this.selectedIndicesSetForRender.has(dataIndex);
    const isPrimarySelection = this.primarySelectedIndex === dataIndex;
    const isReferenceSelection = this.referenceSelectedIndex === dataIndex;
    const isFocused = this.focusedIndex === dataIndex;
    const rowClass = classMap({
      'selected': isSelected,
      'primary-selected': isPrimarySelection,
      'reference-selected': isReferenceSelection,
      'focused': isFocused
    });
    const mouseDown = (e: MouseEvent) => {
      if (!this.selectionEnabled) return;
      this.handleRowClick(e, dataIndex, displayDataIndex);
    };
    const mouseEnter = (e: MouseEvent) => {
      this.handleRowMouseEnter(e, dataIndex);
    };
    const mouseLeave = (e: MouseEvent) => {
      this.handleRowMouseLeave(e, dataIndex);
    };

    // clang-format off
    return html`
      <tr class="${rowClass}" @mousedown=${mouseDown} @mouseenter=${mouseEnter}
        @mouseleave=${mouseLeave}>
        ${data.rowData.map(d => {
          if (d == null) {
            return html`<td></td>`;
          }
          if (typeof d === "string" && d.startsWith(IMAGE_PREFIX)) {
            return html`<td><img class='table-img' src=${d.toString()}></td>`;
          }
          if (isTemplateResult(d) || d.constructor === Object) {
            const templateResult = isTemplateResult(d) ?
              d : (d as SortableTemplateResult).template;
            return html`<td>${templateResult}</td>`;
          }
          const entryClasses = {numeric: isNumber(d)};
          return html`<td><div class="${classMap(entryClasses)}">
              ${formatForDisplay(d, undefined, true)}</div></td>`;
        })}
      </tr>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-data-table': DataTable;
  }
}
