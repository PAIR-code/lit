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
import {html, TemplateResult} from 'lit';
import {customElement, property} from 'lit/decorators';
import {isTemplateResult} from 'lit/directive-helpers';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {action, computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {formatForDisplay} from '../lib/types';
import {isNumber, randInt} from '../lib/utils';

import {styles as sharedStyles} from '../lib/shared_styles.css';
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

/** Wrapper type for column header with optional custom template. */
export interface ColumnHeader {
  name: string;
  html?: TemplateResult;
  rightAlign?: boolean;
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
  @observable.struct @property({type: Array})
  columnNames: Array<string|ColumnHeader> = [];
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

  // Style overrides
  @property({type: Boolean}) verticalAlignMiddle: boolean = false;

  // Callbacks
  @property({type: Object}) onClick: OnPrimarySelectCallback|undefined;
  @property({type: Object}) onHover: OnHoverCallback|undefined;
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
  @observable private readonly columnSearchQueries = new Map<string, string>();
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

  override firstUpdated() {
    const container = this.shadowRoot!.querySelector('.holder')!;
    this.resizeObserver = new ResizeObserver(() => {
      this.adjustEntriesIfHeightChanged();
    });
    this.resizeObserver.observe(container);

    // If inputs changed, re-sort data based on the new inputs.
    this.reactImmediately(() => [this.data, this.rowFilteredData], () => {
      this.stickySortedData = null;
      this.needsEntriesPerPageRecompute = true;
      this.pageNum = 0;
      this.requestUpdate();
    });

    // If sort settings are changed, re-sort data optionally using result of
    // previous sort.
    const triggerSort = () => [this.sortName, this.sortAscending];
    this.reactImmediately(triggerSort, () => {
      this.stickySortedData = this.getSortedData(this.rowFilteredData);
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

    // If pagination is disabled, then ensure all entires can fit on a single
    // page.
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
    const rows: NodeListOf<HTMLElement> =
        this.shadowRoot!.querySelectorAll('tbody > tr');
    let height = 0;
    let i = 0;

    // Iterate over rows, adding up their height until they fill the container,
    // to get the number of rows to display per page.
    for (i = 0; i < rows.length; i++) {
      height += rows[i].getBoundingClientRect().height;
      if (height > availableHeight) {
        this.entriesPerPage = i + 1;
        break;
      }
    }
    if (height === 0) {
      this.entriesPerPage = 10;
    } else if (height <= availableHeight) {
      // If there aren't enough entries to take up the entire container,
      // calculate how many will fill the container based on the heights so far.
      const heightPerEntry = height / i;
      this.entriesPerPage = Math.ceil(availableHeight / heightPerEntry);
    }
    // Round up to the nearest 10.
    this.entriesPerPage = Math.ceil(this.entriesPerPage / 10) * 10;
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
   *   defining sort based on the Score associated with that class). The result
   *   is stored in this.stickySortedData so that future sorts can be "stable"
   *   rather than re-sorting from the input.
   * - displayData is stickySortedData, or rowFilteredData if that is unset.
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
      for (const [key, value] of this.columnSearchQueries) {
        const index = this.columnStrings.indexOf(key);
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
    if (this.stickySortedData != null) {
      return this.stickySortedData;
    } else {
      return this.getSortedData(this.rowFilteredData);
    }
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
    this.sortName = undefined;    // reset to input ordering
    this.showColumnMenu = false;  // hide search bar
    // Reset sticky sort and re-render from input data.
    this.stickySortedData = null;
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
    return html`
      <div class="holder">
        <table>
          <thead>
            ${this.columnHeaders.map(c => this.renderColumnHeader(c))}
          </thead>
          <tbody>
            ${this.pageData.map((d, rowIndex) => this.renderRow(d, rowIndex))}
          </tbody>
          <tfoot>
            ${this.renderFooter()}
          </tfoot>
        </table>
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
      const newPageNum = modPageNumber(this.pageNum - 1);
      this.pageNum = newPageNum;
    };
    const firstPage = () => {
      this.pageNum = 0;
    };
    const firstPageButtonClasses = {
      'icon-button': true,
      'disabled': this.pageNum === 0
    };
    const lastPage = () => {
      this.pageNum = this.totalPages - 1;
    };
    const lastPageButtonClasses = {
      'icon-button': true,
      'disabled': this.pageNum === this.totalPages - 1
    };
    const randomPage = () => {
      const newPageNum = randInt(0, this.totalPages);
      this.pageNum = newPageNum;
    };
    // clang-format off
    return html`
      <tr>
        <td colspan=${this.columnNames.length}>
          <div class="footer">
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
            <mwc-icon class=${classMap(lastPageButtonClasses)}
              @click=${lastPage}>
              last_page
            </mwc-icon>
            <mwc-icon class='icon-button mdi-outlined button-extra-margin'
              @click=${randomPage}>
              casino
            </mwc-icon>
          </div>
        </td>
      </tr>`;
    // clang-format on
  }

  renderColumnHeader(header: ColumnHeader) {
    const title = header.name;

    const handleBackgroundClick = (e: Event) => {
      this.resetView();
    };

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

    const searchText = this.columnSearchQueries.get(title) ?? '';

    const isSearchActive = () => {
      const searchString = this.columnSearchQueries.get(title);
      if (searchString ||
          (this.showColumnMenu && this.columnMenuName === title)) {
        return true;
      }
      return false;
    };

    const searchMenuStyle = styleMap({
      'visibility':
          (this.showColumnMenu && this.columnMenuName === title ? 'visible' :
                                                                  'hidden'),
    });

    const menuButtonStyle =
        styleMap({'outline': (isSearchActive() ? 'auto' : 'none')});

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
          const inputElem = this.shadowRoot!.querySelector(`th#${title} .togglable-menu-holder input`) as HTMLElement;
          inputElem.focus();
        });
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
    const headerClasses =
        classMap({'column-header': true, 'right-align': header.rightAlign!});

    // clang-format off
    return html`
        <th id=${title} @click=${handleBackgroundClick}>
          <div class=${headerClasses} title=${title}>
            <div class="header-holder">
              <div @click=${toggleSort}>${header.html!}</div>
              ${this.searchEnabled ? html`
                <div class="menu-button-container">
                  <mwc-icon class="menu-button" style=${menuButtonStyle}
                   @click=${handleMenuButton}>search</mwc-icon>
                </div>` : null}
              <div class="arrow-container" @click=${toggleSort}>
                <mwc-icon class=${upArrowClasses}>arrow_drop_up</mwc-icon>
                <mwc-icon class=${downArrowClasses}>arrow_drop_down</mwc-icon>
              </div>
            </div>
          </div>
          ${this.searchEnabled ? html`
            <div class='togglable-menu-holder' style=${searchMenuStyle}>
                <input type="search" class='search-input'
                .value=${searchText}
                placeholder="Search" @input=${handleSearchChange}/>
            </div>` : null}
        </th>
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

    const formatCellContents = (d: TableEntry) => {
      if (d == null) return null;

      if (typeof d === 'string' && d.startsWith(IMAGE_PREFIX)) {
        return html`<img class='table-img' src=${d.toString()}>`;
      }
      if (isTemplateResult(d) || d.constructor === Object) {
        const templateResult =
            isTemplateResult(d) ? d : (d as SortableTemplateResult).template;
        return html`${templateResult}`;
      }

      // Text formatting uses pre-wrap, so be sure that this template doesn't
      // add any extra whitespace inside the div.
      // clang-format off
      return html`
          <div class="text-cell">${formatForDisplay(d, undefined, true)}</div>`;
      // clang-format on
    };

    const cellClasses = this.columnHeaders.map(
        h => classMap({'cell-holder': true, 'right-align': h.rightAlign!}));
    const cellStyles = styleMap({
      verticalAlign: this.verticalAlignMiddle ? 'middle' : 'top'
    });
    // clang-format off
    return html`
      <tr class="${rowClass}" @mousedown=${mouseDown} @mouseenter=${mouseEnter}
        @mouseleave=${mouseLeave}>
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
