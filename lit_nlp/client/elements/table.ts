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
import './checkbox';
import './popup_container';

import {ascending, descending} from 'd3';  // array helpers.
import {html, TemplateResult} from 'lit';
import {customElement, property} from 'lit/decorators';
import {isTemplateResult} from 'lit/directive-helpers';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {action, computed, observable} from 'mobx';
import * as papa from 'papaparse';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay} from '../lib/types';
import {isNumber, median, numberRangeFnFromString, randInt} from '../lib/utils';

import {PopupContainer} from './popup_container';
import {styles} from './table.css';

/** Function for supplying table entry template result based on row state. */
export type TemplateResultFn =
    (isSelected: boolean, isPrimarySelection: boolean,
     isReferenceSelection: boolean, isFocused: boolean, isStarred: boolean) =>
        TemplateResult;

type SortableTableEntry = string|number;
/** Wrapper type for sortable custom data table entries */
export interface SortableTemplateResult {
  template: TemplateResult|TemplateResultFn;
  value: SortableTableEntry;
}
/** Wrapper types for the data supplied to the data table */
export type TableEntry = string|number|TemplateResult|SortableTemplateResult;
/** Wrapper types for the data supplied to the data table */
export type TableData = TableEntry[]|{[key: string]: TableEntry};

/** Wrapper type for column header with optional custom template. */
export interface ColumnHeader {
  name: string;
  html?: TemplateResult;
  rightAlign?: boolean;
  /**
   * If vocab provided and search enabled, then the column is searchable
   *  through selected items from the vocab list.
   */
  vocab?: string[];
}

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

/** Type for search filters in data headers. */
export type FilterFn = (entry: SortableTableEntry) => boolean;

/** Info stored for the filter info for each column. */
interface ColumnFilterInfo {
  /** The generated filter function. */
  fn: FilterFn;
  /** String lists backing the user settings of a search filter. */
  values: string[];
}

enum SpanAnchor {
  START,
  END,
}

const IMAGE_PREFIX = 'data:image';

const CONTAINER_HEIGHT_CHANGE_DELTA = 20;
const PAGE_SIZE_INCREMENT = 10;

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
  @observable.struct @property({type: Array})
      columnNames: Array<string|ColumnHeader> = [];
  @observable.struct @property({type: Array}) selectedIndices: number[] = [];
  @observable.struct @property({type: Array}) starredIndices: number[] = [];
  @observable @property({type: Number}) primarySelectedIndex: number = -1;
  @observable @property({type: Number}) referenceSelectedIndex: number = -1;
  // TODO(lit-dev): consider a custom reaction to make this more responsive,
  // instead of triggering a full re-render.
  @observable @property({type: Number}) focusedIndex: number = -1;

  // Mode controls
  @observable @property({type: Boolean}) selectionEnabled: boolean = false;
  @observable @property({type: Boolean}) searchEnabled: boolean = false;
  @observable @property({type: Boolean}) paginationEnabled: boolean = false;
  @observable @property({type: Boolean}) exportEnabled: boolean = false;

  /** Lowest row index of the continguous (i.e., shift-click) selection. */
  @property({type: Number}) shiftSelectionStartIndex: number = 0;
  /** Highest row index of the continguous (i.e., shift-click) selection. */
  @property({type: Number}) shiftSelectionEndIndex: number = 0;

  // Style overrides
  @property({type: Boolean}) verticalAlignMiddle: boolean = false;

  // Callbacks
  @property({type: Object}) onClick?: OnPrimarySelectCallback;
  @property({type: Object}) onHover?: OnHoverCallback;
  @property({type: Object}) onSelect: OnSelectCallback = () => {};
  @property({type: Object}) onPrimarySelect: OnPrimarySelectCallback = () => {};

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Sort order precedence: 1) sortName, 2) input order
  @observable @property({type: String}) sortName?: string;
  @observable @property({type: Boolean}) sortAscending = true;
  @observable private showColumnMenu = false;
  @observable private columnMenuName = '';
  // Filters for each column when search is used.
  @observable
  private readonly columnFilterInfo = new Map<string, ColumnFilterInfo>();
  @observable private pageNum = 0;
  @observable private entriesPerPage = PAGE_SIZE_INCREMENT;

  @property({type: String}) downloadFilename: string = 'data.csv';

  private readonly resizeObserver = new ResizeObserver(() => {
    this.adjustEntriesIfHeightChanged();
  });
  private needsEntriesPerPageRecompute = true;
  private lastContainerHeight = 0;

  private selectedIndicesSetForRender = new Set<number>();
  private shiftSpanAnchor = SpanAnchor.START;
  private hoveredIndex: number|null = null;

  // Timeout to not spam hover events as a mouse moves over the table.
  private readonly HOVER_TIMEOUT_MS = 3;
  private hoverTimeoutId: number|null = null;

  override firstUpdated() {
    const container = this.shadowRoot!.querySelector('.holder')!;
    this.resizeObserver.observe(container);

    // If inputs changed, re-sort data based on the new inputs.
    this.reactImmediately(() => [this.data, this.rowFilteredData], () => {
      this.needsEntriesPerPageRecompute = true;
      this.requestUpdate();
    });

    // Reset page number if invalid on change in total pages.
    this.reactImmediately(() => this.totalPages, () => {
      const isPageOverflow = this.pageNum >= this.totalPages;
      if (isPageOverflow) {this.pageNum = 0;}

      const lastIndexOnPage = this.entriesPerPage * (this.pageNum + 1);
      const isHoveredInvisible =
          this.hoveredIndex != null && this.hoveredIndex > lastIndexOnPage;
      if (isHoveredInvisible) {this.hoveredIndex = null;}
    });
  }

  // tslint:disable-next-line:no-any
  override shouldUpdate(changedProperties: any) {
    if (changedProperties.get('data')) {
      // Let's just punt on the issue of maintaining shift selection behavior
      // when the data changes (via filtering, for example)
      this.shiftSelectionStartIndex = 0;
      this.shiftSelectionEndIndex = 0;
    }
    return true;
  }

  override updated() {
    if (this.needsEntriesPerPageRecompute) {
      this.computeEntriesPerPage();
    }
  }

  private getContainerHeight() {
    const container: HTMLElement =
        // tslint:disable-next-line:no-unnecessary-type-assertion
        this.shadowRoot!.querySelector('.holder')! as HTMLElement;
    return container.getBoundingClientRect().height;
  }

  // If the row container's height has changed significantly, then recompute
  // entries per row.
  private adjustEntriesIfHeightChanged() {
    const containerHeight = this.getContainerHeight();
    if (Math.abs(containerHeight - this.lastContainerHeight) >
        CONTAINER_HEIGHT_CHANGE_DELTA) {
      this.lastContainerHeight = containerHeight;
      this.computeEntriesPerPage();
    }
  }

  private computeEntriesPerPage() {
    this.needsEntriesPerPageRecompute = false;

    // If pagination is disabled, then put all entries on a single page.
    if (!this.paginationEnabled) {
      this.entriesPerPage = this.rowFilteredData.length + 1;
      return;
    }

    const containerHeight = this.getContainerHeight();
    // Account for the height of the header and footer.
    const headerHeight =
        this.shadowRoot!.querySelector('thead')!.getBoundingClientRect().height;
    const footerHeight =
        this.shadowRoot!.querySelector('tfoot')!.getBoundingClientRect().height;
    const availableHeight =
        Math.max(0, containerHeight - headerHeight - footerHeight);

    // Iterate over rows, adding up their heights until they overfill the
    // container (or run out of rows), to get the number of rows per page.
    let height = 0;
    let i = 0;
    const rows: NodeListOf<HTMLElement> =
        this.shadowRoot!.querySelectorAll('tbody > tr.lit-data-table-row');

    // This function computes entriesPerPage assuming that the current page is
    // full of rows that it can use to compute entriesPerPage based on the
    // available page height. The only time that isn't true is if the user is
    // one the last page of results, in which case computing a new
    // entriesPerPage value may result in an incorrect computation, so bail out.
    if (rows.length < this.entriesPerPage) {return;}

    while (height < availableHeight && i < rows.length) {
      height += rows[i].getBoundingClientRect().height;
      i += 1;
    }

    // Calculate how many rows will fit on a page given the available space.
    let entriesPerPage;
    if (height === 0 || i === 0) {
      // No content, calculate how many would fill it assuming minimum row size
      // given styles, 28px.
      // div.cell-holder (18px height + 8px padding) + td (2px padding) = 28px
      entriesPerPage = Math.ceil(availableHeight / 28);
    } else {
      const hasFillers =
          this.shadowRoot!.querySelectorAll('tbody > tr.filler').length > 0;

      if (hasFillers || height === availableHeight) {
        // The height is exactly the same as the container, which means either
        // LIT got incredibly lucky (unlikely) or that the browser's table
        // algorithm is stretching the rows to fill the available space (very
        // likely). Since it's so likely that the table-row elements are being
        // stretched to fit the available space, use the median div height
        // instead of mean row height to calculate how many would fit.
        const cellHolderSelector = 'tbody > tr > td > div.cell-holder';
        const cellHolders: NodeListOf<HTMLDivElement> =
            this.shadowRoot!.querySelectorAll(cellHolderSelector);
        const cellHolderHeights = [...cellHolders.values()].map(div =>
            div.getBoundingClientRect().height);
        const medianHeight = median(cellHolderHeights);
        entriesPerPage = Math.ceil(availableHeight / medianHeight);
      } else if (height < availableHeight) {
        // If there aren't enough entries to fill the container, calculate how
        // many will fill the container based on the mean height.
        const meanHeight = height / i;
        entriesPerPage = Math.ceil(availableHeight / meanHeight);
      } else {
        // There are enough entries to fill the container, use i.
        entriesPerPage = i;
      }
    }

    // Round up to the nearest 10.
    this.entriesPerPage =
        Math.ceil(entriesPerPage / PAGE_SIZE_INCREMENT) * PAGE_SIZE_INCREMENT;
  }

  private getSortableEntry(colEntry: TableEntry): SortableTableEntry {
    // Passthrough values if TableEntry is number or string. If it is
    // TemplateResult return 0 for sorting purposes. If it is a sortable
    // tempate result then sort by the underlying sortable value.
    if (typeof colEntry === 'string' || isNumber(colEntry)) {
      return colEntry as SortableTableEntry;
    }
    if (isTemplateResult(colEntry)) {
      return 0;
    }
    return (colEntry as SortableTemplateResult).value;
  }

  @computed
  get sortIndex(): number|undefined {
    return this.sortName != null ? this.columnStrings.indexOf(this.sortName) :
                                   undefined;
  }

  private shouldRightAlignColumn(index: number) {
    const values = this.pageData.map(row => row.rowData[index]);
    const numericValues = values.filter(d => isNumber(d));
    if (numericValues.length / values.length > 0.5) {
      return true;
    }
    return false;
  }


  /**
   * Column names. To avoid cyclical dependencies, this needs to be independent
   * of columnHeaders, since the names are used to select the table data,
   * which is in turn used to compute some formatting defaults.
   */
  @computed
  get columnStrings(): string[] {
    return this.columnNames.map(
        colInfo => (typeof colInfo === 'string') ? colInfo : colInfo.name);
  }

  @computed
  get columnHeaders(): ColumnHeader[] {
    return this.columnNames.map((colInfo, index) => {
      const header: ColumnHeader = (typeof colInfo === 'string') ?
          {name: colInfo} :
          {...colInfo};
      header.html =
          header.html ?? html`<div class="header-text">${header.name}</div>`;
      header.rightAlign =
          header.rightAlign ?? this.shouldRightAlignColumn(index);
      return header;
    });
  }

  /**
   * First pass processing input data to canonical form.
   * The data goes through several stages before rendering:
   * - this.data is the input data (bound via element property)
   * - indexedData converts to parallel-list form and adds input indices
   *   for later reference
   * - rowFilteredData filters to a subset of rows, based on search criteria
   * - getSortedData() is used to sort the data if 1) the user has enabled sort-
   *   by-column by clicking a column header, or 2) a DataTable consumer wants
   *   to impose a context-specific sort behavior (e.g., ClassificationModule
   *   defining sort based on the Score associated with that class).
   * - displayData is uses sorted rowFilteredData.
   *   This is used to actually render the table.
   */
  @computed
  get indexedData(): TableRowInternal[] {
    // Convert any objects to simple arrays by selecting fields.
    const convertedData: TableEntry[][] = this.data.map((d: TableData) => {
      if (d instanceof Array) return d;
      return this.columnStrings.map(k => d[k]);
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
      for (const [key, info] of this.columnFilterInfo) {
        const index = this.columnStrings.indexOf(key);
        if (index === -1) return;

        const col = this.getSortableEntry(item.rowData[index]);
        isShownByTextFilter = isShownByTextFilter && info.fn(col);
      }
      return isShownByTextFilter;
    });
  }

  getSortedData(source: TableRowInternal[]): TableRowInternal[] {
    if (this.sortName != null) {
      const sorter = this.sortAscending ? ascending : descending;

      return source.slice().sort((a, b) => sorter(
        this.getSortableEntry(a.rowData[this.sortIndex!]),
        this.getSortableEntry(b.rowData[this.sortIndex!])));
    }

    return source;
  }

  @computed
  get displayData(): TableRowInternal[] {
    return this.getSortedData(this.rowFilteredData);
  }

  getArrayData(): SortableTableEntry[][] {
    // displayData is the visible data for all pages.
    return this.displayData.map(
        (row: TableRowInternal) => row.rowData.map(this.getSortableEntry));
  }

  getCSVContent(): string {
    return papa.unparse(
        {fields: this.columnStrings, data: this.getArrayData()},
        {newline: '\r\n'});
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
  get fillerRows(): TemplateResult[] {
    const rows: TemplateResult[] = [];
    for(let i = 0; i < (this.entriesPerPage - this.pageData.length); i++) {
      rows.push(html`<tr class="filler">
        <td colspan=${this.columnNames.length}>&nbsp;</td>
      </tr>`);
    }
    return rows;
  }

  @computed
  get isPaginated(): boolean {
    return this.paginationEnabled &&
        (this.displayData.length > this.entriesPerPage);
  }

  @computed
  get hasFooter(): boolean {
    return this.isPaginated || this.exportEnabled;
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
      this.setPrimarySelectedIndex(rowIndex);
    }
    //  Rather complicated logic for handling shift-click
    else if (e.shiftKey) {
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

      this.setPrimarySelectedIndex(rowIndex);
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
    if (this.hoverTimeoutId != null) {
      window.clearTimeout(this.hoverTimeoutId);
    }

    this.hoverTimeoutId = window.setTimeout(() => {
      this.hoveredIndex = dataIndex;
      if (this.onHover != null) {
        this.onHover(this.hoveredIndex);
      }
    }, this.HOVER_TIMEOUT_MS);
  }

  private handleRowMouseLeave(e: MouseEvent, dataIndex: number) {
    if (this.hoverTimeoutId != null) {
      window.clearTimeout(this.hoverTimeoutId);
    }

    this.hoverTimeoutId = window.setTimeout(() => {
      if (dataIndex === this.hoveredIndex) {
        this.hoveredIndex = null;
      }

      if (this.onHover != null) {
        this.onHover(this.hoveredIndex);
      }
    }, this.HOVER_TIMEOUT_MS);
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
    return this.sortName === undefined && this.columnFilterInfo.size === 0;
  }

  @computed
  get isFiltered() {
    return this.columnFilterInfo.size !== 0;
  }

  resetView() {
    this.columnFilterInfo.clear();
    this.sortName = undefined;    // reset to input ordering
    this.showColumnMenu = false;  // hide search bar
    this.requestUpdate();
  }

  getVisibleDataIdxs(): number[] {
    return this.displayData.map(d => d.inputIndex);
  }

  override render() {
    // Make a private, temporary set of selectedIndices to simplify lookup
    // in the row render method
    this.selectedIndicesSetForRender = new Set<number>(this.selectedIndices);

    // clang-format off
    return html`<div class="holder">
      <table class=${classMap({'has-footer': this.hasFooter})}>
        <thead>
          ${this.columnHeaders.map((c, i) =>
              this.renderColumnHeader(c, i === this.columnHeaders.length - 1))}
        </thead>
        <tbody>
          ${this.pageData.map((d, rowIndex) => this.renderRow(d, rowIndex))}
          ${this.hasFooter ? this.fillerRows : null}
        </tbody>
        <tfoot>
          ${this.renderFooter()}
        </tfoot>
      </table>
    </div>`;
    // clang-format on
  }

  renderPaginationControls() {
    // Use this modulo function so that if pageNum is negative (when
    // decrementing pages), the modulo returns the expected positive value.
    const modPageNumber = (pageNum: number) => {
      return ((pageNum % this.totalPages) + this.totalPages) % this.totalPages;
    };

    const changePage = (offset: number) => {
      this.pageNum = modPageNumber(this.pageNum + offset);
    };
    const firstPage = () => {
      this.pageNum = 0;
    };
    const lastPage = () => {
      this.pageNum = this.totalPages - 1;
    };
    const randomPage = () => {
      this.pageNum = randInt(0, this.totalPages);
    };

    const firstPageButtonClasses = {
      'icon-button': true,
      'disabled': this.pageNum === 0
    };
    const lastPageButtonClasses = {
      'icon-button': true,
      'disabled': this.pageNum === (this.totalPages - 1)
    };

    // clang-format off
    return html`
      <mwc-icon class=${classMap(firstPageButtonClasses)}
        @click=${firstPage}>
        first_page
      </mwc-icon>
      <mwc-icon class='icon-button'
        @click=${() => {changePage(-1);}}>
        chevron_left
      </mwc-icon>
      <div>
       Page
       <span class="current-page-num">${this.pageNum + 1}</span>
       of ${this.totalPages}
      </div>
      <mwc-icon class='icon-button'
         @click=${() => {changePage(1);}}>
        chevron_right
      </mwc-icon>
      <mwc-icon class=${classMap(lastPageButtonClasses)}
        @click=${lastPage}>
        last_page
      </mwc-icon>
      <mwc-icon class='icon-button mdi-outlined'
        title="Go to a random page" @click=${randomPage}>
        casino
      </mwc-icon>
    `;
    // clang-format on
  }

  renderExportControls() {
    const copyCSV = () => {
      const csvContent = this.getCSVContent();
      navigator.clipboard.writeText(csvContent);
    };

    const downloadCSV = () => {
      const csvContent = this.getCSVContent();
      const blob = new Blob([csvContent], {type: 'text/csv'});
      const a = window.document.createElement('a');
      a.href = window.URL.createObjectURL(blob);
      a.download = this.downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      const controls: PopupContainer =
          this.shadowRoot!.querySelector('popup-container.download-popup')!;
      controls.expanded = false;
    };

    const updateFilename = (e: Event) => {
      // tslint:disable-next-line:no-any
      this.downloadFilename = (e as any).target.value as string;
    };

    function onEnter(e: KeyboardEvent) {
      if (e.key === 'Enter') downloadCSV();
    }

    // clang-format off
    return html`
      <mwc-icon class='icon-button'
        title="Copy ${this.displayData.length} rows as CSV"
        @click=${copyCSV}>
        file_copy
      </mwc-icon>
      <popup-container class='download-popup'>
        <mwc-icon class='icon-button' slot='toggle-anchor'
          title="Download ${this.displayData.length} rows as CSV">
          file_download
        </mwc-icon>
        <div class='download-popup-controls'>
          <label for="filename">Filename</label>
          <input type="text" name="filename" value=${this.downloadFilename}
           @input=${updateFilename} @keydown=${onEnter}>
          <button class='filled-button nowrap' @click=${downloadCSV}
            ?disabled=${!this.downloadFilename}>
            Download ${this.displayData.length} rows
          </button>
        </div>
      </popup-container>
    `;
    // clang-format on
  }

  renderFooter() {
    if (!this.hasFooter) return null;

    // clang-format off
    return html`
      <tr>
        <td colspan=${this.columnNames.length}>
          <div class="footer">
            ${this.isPaginated ? this.renderPaginationControls() : null}
            <div class='footer-spacer'></div>
            ${this.exportEnabled ? this.renderExportControls() : null}
          </div>
        </td>
      </tr>`;
    // clang-format on
  }

  private columnNameToId(name: string) {
    return name.replace(/\s+/g, '');
  }

  renderColumnHeader(header: ColumnHeader, isRightmostHeader: boolean) {
    const title = header.name;
    const headerId = this.columnNameToId(title);

    const toggleSort = (e: Event) => {
      e.stopPropagation();

      // Table supports three sort states/transitions after a click:
      if (this.sortName !== title) {
        //   1. If title !== sortName, sort by that title in ascending order
        this.sortName = title;
        this.sortAscending = true;
      } else {
        if (this.sortAscending) {
          // 2. If title === sortName && ascending, switch to descending
          this.sortAscending = false;
        } else {
          // 3. If title === sortName && descending, turn off sort
          this.sortName = undefined;
        }
      }
    };

    const isSearchActive = () => {
      const searchValues = this.columnFilterInfo.get(title)?.values;
      const hasSearchValues =
          searchValues && searchValues.length > 0 && searchValues[0].length > 0;
      const {showColumnMenu, columnMenuName} = this;
      return hasSearchValues || (showColumnMenu && columnMenuName === title);
    };

    const handleMenuButton = (e: Event) => {
      e.stopPropagation();
      if (this.columnMenuName === title) {
        this.showColumnMenu = !this.showColumnMenu;
      } else {
        this.columnMenuName = title;
        this.showColumnMenu = true;
      }
      // Focus cursor on newly-shown input box.
      if (this.showColumnMenu) {
        window.requestAnimationFrame(() => {
          const inputElem = this.shadowRoot!.querySelector(
              `th#${headerId} .togglable-menu-holder input`) as HTMLElement;
          if (inputElem != null) {
            inputElem.focus();
          }
        });
      }
    };

    const isUpActive = this.sortName === title && this.sortAscending;
    const isUpInactive = this.sortName === title && !this.sortAscending;
    const isDownActive = this.sortName === title && !this.sortAscending;
    const isDownInactive = this.sortName === title && this.sortAscending;

    const arrowStyle =
        styleMap((isDownActive || isUpActive) ? {'display': 'block'} : {});
    const menuButtonStyle = styleMap(isSearchActive() ?
        {'display': 'block', 'outline': 'auto'} : {});

    const upArrowClasses = classMap({
      'arrow': true,
      'up': true,
      'active': isUpActive,
      'inactive': isUpInactive,
    });
    const downArrowClasses = classMap({
      'arrow': true,
      'down': true,
      'active': isDownActive,
      'inactive': isDownInactive,
    });
    const headerClasses =
        classMap({'column-header': true, 'right-align': header.rightAlign!});

    // clang-format off
    return html`
        <th id=${headerId}>
          <div class=${headerClasses} title=${title}>
            <div class="header-holder">
              <div @click=${toggleSort}>${header.html!}</div>
              ${this.searchEnabled ? html`
                <div class="menu-button-container">
                  <mwc-icon class="menu-button" style=${menuButtonStyle}
                   title="Filter" @click=${handleMenuButton}>
                   filter_alt
                  </mwc-icon>
                </div>` : null}
              <div class="arrow-container" @click=${toggleSort}>
                <mwc-icon class=${upArrowClasses} style=${arrowStyle}
                  title="Sort (ascending)">
                  arrow_drop_up
                </mwc-icon>
                <mwc-icon class=${downArrowClasses} style=${arrowStyle}
                  title="Sort (descending)">
                  arrow_drop_down
                </mwc-icon>
              </div>
            </div>
          </div>
          ${this.searchEnabled ?
              this.renderSearch(header, isRightmostHeader) : null}
        </th>
      `;
    // clang-format on
  }

  renderSearch(header: ColumnHeader, isRightmostHeader: boolean) {
    const searchMenuStyle = styleMap({
      'display': (
          this.showColumnMenu && this.columnMenuName === header.name ? 'block' :
                                                                       'none'),
    });
    const searchMenuClasses = classMap({
      'togglable-menu-holder': true,
      'checkbox-holder': header.vocab != null,
      'right-aligned-search': isRightmostHeader
    });

    if (header.vocab == null) {
      // If the column has no vocab, then set filter through a free-Text
      // field with different behavior for string columns vs numeric columns.
      const handleSearchChange = (e: KeyboardEvent) => {
        const searchQuery = (e.target as HTMLInputElement)?.value || '';
        function fn(col: SortableTableEntry) {
          if (typeof col === 'string') {
            // String columns use a reg ex search.
            return col.search(new RegExp(searchQuery)) !== -1;
          } else if (typeof col === 'number') {
            // Numeric columns use a range-based search.
            const matchFn = numberRangeFnFromString(searchQuery);
            return matchFn(col);
          } else {
            return false;
          }
        }
        this.columnFilterInfo.set(header.name, {fn, values: [searchQuery]});
      };

      const searchText = this.columnFilterInfo.has(header.name) ?
          this.columnFilterInfo.get(header.name)!.values[0] :
          '';

      return html`
        <div class=${searchMenuClasses} style=${searchMenuStyle}>
            <input type="search" class='search-input'
            .value=${searchText}
            placeholder="Filter" @input=${handleSearchChange}/>
        </div>`;  // clang-format off
    } else {
      // For columns with vocabs, use a set of checkboxes, one per each vocab
      // item.
      return html`
        <div class=${searchMenuClasses} style=${searchMenuStyle}>
          ${header.vocab.map(option => {
            const isChecked = this.columnFilterInfo.get(
              header.name)?.values.includes(option) || false;
            // tslint:disable-next-line:no-any
            const handleCheck = (e: any) => {
              let list = this.columnFilterInfo.get(header.name)?.values || [];
              if (e.target.checked) {
                list.push(option);
              } else {
                list = list.filter(name => name !== option);
              }

              const fn = (col: SortableTableEntry) => {
                const checkedItems = this.columnFilterInfo.get(
                  header.name)?.values || [];
                // If no items are checked, do not filter the column.
                if (checkedItems.length === 0) {
                  return true;
                }
                // If any items are checked, only show the items that match
                // one of the checked vocab values.
                return checkedItems.includes(col.toString());
              };
              this.columnFilterInfo.set(header.name, {fn, values: list});
            };
            return html`<lit-checkbox label=${option} ?checked=${isChecked}
                @change=${handleCheck}></lit-checkbox>`;
        })}
        </div>`;  // clang-format off
    }
  }

  renderRow(data: TableRowInternal, rowIndex: number) {
    const dataIndex = data.inputIndex;
    const displayDataIndex = rowIndex + this.pageNum * this.entriesPerPage;

    const isSelected = this.selectedIndicesSetForRender.has(dataIndex);
    const isPrimarySelection = this.primarySelectedIndex === dataIndex;
    const isReferenceSelection = this.referenceSelectedIndex === dataIndex;
    const isFocused = this.focusedIndex === dataIndex;
    const isStarred = this.starredIndices.indexOf(dataIndex) !== -1;
    const rowClass = classMap({
      'lit-data-table-row': true,
      'selected': isSelected,
      'primary-selected': isPrimarySelection,
      'reference-selected': isReferenceSelection,
      'focused': isFocused
    });
    const onClick = (e: MouseEvent) => {
      if (!this.selectionEnabled) return;
      this.handleRowClick(e, dataIndex, displayDataIndex);
    };
    const mouseEnter = (e: MouseEvent) => {
      this.handleRowMouseEnter(e, dataIndex);
    };
    const mouseLeave = (e: MouseEvent) => {
      this.handleRowMouseLeave(e, dataIndex);
    };

    // Prevent text highlighting when using shift+click to select rows.
    const mouseDown = (e: MouseEvent) => {
      if (e.shiftKey && this.selectionEnabled) {
        e.preventDefault();
      }
    };

    function formatCellContents(d: TableEntry) {
      if (d == null) return null;

      if (typeof d === 'string' && d.startsWith(IMAGE_PREFIX)) {
        return html`<img class='table-img' src=${d.toString()}>`;
      }
      if (isTemplateResult(d)) {
        return d;
      }
      if (d.constructor === Object) {
        let templateResult = null;
        const t = (d as SortableTemplateResult).template;
        if (typeof t === 'function') {
          templateResult = t(
            isSelected,
            isPrimarySelection,
            isReferenceSelection,
            isFocused,
            isStarred);
        } else {
          templateResult = t;
        }
        return html`${templateResult}`;
      }

      // Text formatting uses pre-wrap, so be sure that this template
      // doesn't add any extra whitespace inside the div.
      // clang-format off
      return html`
          <div class="text-cell">${formatForDisplay(d, undefined, true)}</div>`;
      // clang-format on
    }

    const cellClasses = this.columnHeaders.map(
        h => classMap({'cell-holder': true, 'right-align': h.rightAlign!}));
    const cellStyles = styleMap(
        {'vertical-align': this.verticalAlignMiddle ? 'middle' : 'top'});
    // clang-format off
    return html`
      <tr class="${rowClass}" @click=${onClick} @mouseenter=${mouseEnter}
        @mouseleave=${mouseLeave} @mousedown=${mouseDown}>
        ${data.rowData.map((d, i) =>
            html`<td style=${cellStyles}><div class=${cellClasses[i]}>${
              formatCellContents(d)
            }</div></td>`)}
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
