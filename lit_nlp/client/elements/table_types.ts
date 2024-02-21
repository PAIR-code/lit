/**
 * @license
 * Copyright 2022 Google LLC
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
 * Types used in the data table component
 */

import {TemplateResult} from 'lit';

/** Function for supplying table entry template result based on row state. */
export type TemplateResultFn =
    (isSelected: boolean, isPrimarySelection: boolean,
     isReferenceSelection: boolean, isFocused: boolean, isStarred: boolean) =>
        TemplateResult;

/** Wrapper type for sortable entry tables */
export type SortableTableEntry = string|number;

/** Wrapper type for sortable custom data table entries */
export interface SortableTemplateResult {
  template: TemplateResult|TemplateResultFn;
  value: SortableTableEntry;
}
/** Wrapper types for the data supplied to the data table */
export type TableEntry = string|number|string[]|TemplateResult|SortableTemplateResult;

/** Wrapper types for the rows of data supplied to the data table */
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
  /** The maximum width of the column, in px. */
  maxWidth?: number;
  /** The minimum width of the column, in px. */
  minWidth?: number;
  /** The width of the column, in px. */
  width?: number;
  /**
   * If defined, the table will provide a LitTooltip for this header and pass
   * this value to that tooltip via its content= attribute.
   */
  tooltip?: string;
  /** A value (in px) passed to --tooltip-width via the styles= attr. */
  tooltipWidth?: number;
  /** A value (in px) passed to --tooltip-max-width via the styles= attr. */
  tooltipMaxWidth?: number;
}

/** Internal data, including metadata */
export interface TableRowInternal {
  inputIndex: number; /* index in original this.data */
  rowData: TableEntry[];
}


