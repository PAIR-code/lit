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

import '@material/mwc-icon';

import * as d3 from 'd3';
import {html, LitElement} from 'lit';
// tslint:disable:no-new-decorators
import {customElement, property} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed, observable} from 'mobx';

import {MAJOR_TONAL_COLORS, ramp} from '../lib/colors';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './data_matrix.css';


// Custom color ramp for the Data Matrix
const LOW = 0, HIGH = 8;  // 0: -50, 1: -100, 2: -200, etc., HIGH gets excluded
// Text color flips (black => white) above -600, calc the % where that happens
const COLOR_FLIP_PCT = Math.floor((6 - LOW) / (HIGH - 1 - LOW) * 100);
const COLOR_RAMP = ramp([...MAJOR_TONAL_COLORS.primary.slice(LOW, HIGH)
                                                      .map(c => c.color)]);

/**
 * Stores information for each confusion matrix cell.
 */
export interface MatrixCell {
  size: number;
  selected: boolean;
}

/** Custom selection event interface for DataMatrix */
export interface MatrixSelection {
  // A list of the cells selected in the data matrix, by row and column index.
  cells: Array<[number, number]>;
}

/**
 * An element that displays a confusion-matrix style vis for datapoints.
 */
@customElement('data-matrix')
export class DataMatrix extends LitElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  @observable verticalColumnLabels = false;
  @property({type: Boolean}) hideEmptyLabels = false;
  @property({type: Array}) matrixCells: MatrixCell[][] = [];
  @property({type: String}) colTitle = "";
  @property({type: String}) rowTitle = "";
  @property({type: Array}) colLabels: string[] = [];
  @property({type: Array}) rowLabels: string[] = [];

  @computed get totalIds(): number {
    let totalIds = 0;
    for (const row of this.matrixCells) {
      for (const cell of row) {
        totalIds += cell.size;
      }
    }
    return totalIds;
  }

  @computed
  get colorScale() {
    // Returns a D3 sequential scale with a domain from 0 (i.e., no selected
    // datapoints are in this cell) to totalIds (i.e., all selected datapoints
    // are in this cell).
    // See https://github.com/d3/d3-scale#sequential-scales
    return d3.scaleSequential(COLOR_RAMP).domain([0, this.totalIds]);
  }

  private updateSelection() {
    const selectedCells: Array<[number, number]> = [];
    for (let i = 0; i < this.matrixCells.length; i++) {
      for (let j = 0; j < this.matrixCells[i].length; j++) {
        const cellInfo = this.matrixCells[i][j];
        if (cellInfo.selected) {
          selectedCells.push([i, j]);
        }
      }
    }
    const event = new CustomEvent<MatrixSelection>('matrix-selection', {
      detail: {
        cells: selectedCells,
      }
    });
    this.dispatchEvent(event);
  }

  private renderColHeader(
      label: string, colIndex: number, colsWithNonZeroCounts: Set<string>) {
    const onColClick = () => {
      const cells = this.matrixCells.map((cells) => cells[colIndex]);
      const allSelected = cells.every((cell) => cell.selected);
      for (const cell of cells) {
        cell.selected = !allSelected;
      }
      this.updateSelection();
      this.requestUpdate();
    };
    if (this.hideEmptyLabels && !colsWithNonZeroCounts.has(label)) {
      return null;
    }
    const classes = classMap({
      'header-cell': true,
      'align-bottom': this.verticalColumnLabels,
      'label-vertical': this.verticalColumnLabels
    });
    // clang-format off
    return html`
      <th class=${classes} @click=${onColClick}>
        <div>${label}</div>
      </th>
    `;
    // clang-format on
  }

  // Render a clickable confusion matrix cell.
  private renderCell(
      rowIndex: number, colIndex: number, colsWithNonZeroCounts: Set<string>) {
    if (this.matrixCells[rowIndex]?.[colIndex] == null) {
      return null;
    }
    const cellInfo = this.matrixCells[rowIndex][colIndex];
    const onCellClick = () => {
      cellInfo.selected = !cellInfo.selected;
      this.updateSelection();
      this.requestUpdate();
    };
    if (this.hideEmptyLabels &&
        !colsWithNonZeroCounts.has(this.colLabels[colIndex])) {
      return null;
    }
    const backgroundColor = this.colorScale(cellInfo.size);
    const percentage = cellInfo.size / this.totalIds * 100;
    const textColor = percentage > COLOR_FLIP_PCT ? 'white' : 'black';
    const border = '2px solid';
    const borderColor =
        cellInfo.selected ? 'var(--lit-cyea-400)' : 'transparent';
    const cellStyle = styleMap({
      'background': `${backgroundColor}`,
      'color': `${textColor}`,
      'border': border,
      'border-color': borderColor,
    });
    return html`
        <td class="cell" style=${cellStyle} @click=${onCellClick}>
          <div class="cell-container">
            <div class="percentage">${percentage.toFixed(1)}%</div>
            <div class="val">(${cellInfo.size})</div>
          </div>
        </td>`;
  }

  // Render a row of the confusion matrix, starting with the clickable
  // row header.
  private renderRow(
      rowLabel: string, rowIndex: number, rowsWithNonZeroCounts: Set<string>,
      colsWithNonZeroCounts: Set<string>) {
    const onRowClick = () => {
      const cells = this.matrixCells[rowIndex];
      const allSelected = cells.every((cell) => cell.selected);
      for (const cell of cells) {
        cell.selected = !allSelected;
      }
      this.updateSelection();
      this.requestUpdate();
    };
    if (this.hideEmptyLabels && !rowsWithNonZeroCounts.has(rowLabel)) {
      return null;
    }
    let totalRowIds = 0;
    for (const cell of this.matrixCells[rowIndex]) {
      totalRowIds += cell.size;
    }
    // clang-format off
    return html`
      <tr>
        <th class="header-cell align-right" @click=${onRowClick}>
          ${rowLabel}
        </th>
        ${this.colLabels.map(
            (colLabel, colIndex) => this.renderCell(
                rowIndex, colIndex,colsWithNonZeroCounts))}
        ${this.renderTotalCell(totalRowIds)}
      </tr>`;
    // clang-format on
  }

  private renderTotalCell(num: number) {
    const percentage = (num / this.totalIds * 100).toFixed(1);
    return html`
        <td class="total-cell">
          <div class="cell-container">
            <div class="percentage">${percentage}%</div>
            <div class="val">(${num})</div>
          </div>
        </td>`;
  }

  private renderColTotalCell(colIndex: number) {
    let totalColIds = 0;
    for (const row of this.matrixCells) {
      totalColIds += row[colIndex].size;
    }
    return this.renderTotalCell(totalColIds);
  }

  private renderTotalRow(colsWithNonZeroCounts: Set<string>) {
    // clang-format off
    return html`
      <tr>
        <th class="total-title-cell align-right">Total</th>
        ${this.colLabels.map(
            (colLabel, colIndex) => {
              if (this.hideEmptyLabels &&
                  !colsWithNonZeroCounts.has(this.colLabels[colIndex])) {
                return null;
              }
              return this.renderColTotalCell(colIndex);
            })}
        <td class="total-cell"></td>
      </tr>`;
    // clang-format on
  }

  private renderColumnRotateButton() {
    const toggleVerticalColumnLabels = () => {
      this.verticalColumnLabels = !this.verticalColumnLabels;
      this.requestUpdate();
    };

    // clang-format off
    return html`
      <mwc-icon class="icon-button"
        title="Rotate column labels"
        @click="${toggleVerticalColumnLabels}">
        ${this.verticalColumnLabels ? 'text_rotate_up' : 'text_rotation_none'}
      </mwc-icon>
    `;
    // clang-format on
  }

  override render() {
    if (this.matrixCells.length === 0) {
      return null;
    }

    const rowsWithNonZeroCounts = new Set<string>();
    const colsWithNonZeroCounts = new Set<string>();
    for (let rowIndex = 0; rowIndex < this.matrixCells.length; rowIndex++) {
      const row = this.matrixCells[rowIndex];
      for (let colIndex = 0; colIndex < row.length; colIndex++) {
        const cell = row[colIndex];
        if (cell.size > 0) {
          rowsWithNonZeroCounts.add(this.rowLabels[rowIndex]);
          colsWithNonZeroCounts.add(this.colLabels[colIndex]);
        }
      }
    }

    const totalColumnClasses = classMap(
        {'total-title-cell': true, 'align-bottom': this.verticalColumnLabels});

    const colsLabelSpan = this.hideEmptyLabels ? colsWithNonZeroCounts.size :
                                                 this.colLabels.length;
    // Add 2 to the appropriate row count to account for the header rows
    // above and below the data rows in the matrix.
    const rowsLabelSpan = (this.hideEmptyLabels ? rowsWithNonZeroCounts.size :
                                                  this.rowLabels.length) + 2;

    // clang-format off
    return html`
      <table>
        <tr>
          <th>${this.renderColumnRotateButton()}</th>
          <td></td>
          <td class='axis-title' colspan=${colsLabelSpan}>
            ${this.colTitle}
          </td>
          <th></th>
        </tr>
        <tr>
          <td class='axis-title label-vertical' rowspan=${rowsLabelSpan}>
            <div>${this.rowTitle}</div>
          </td>
          <td></td>
          ${this.colLabels.map(
              (colLabel, colIndex) => this.renderColHeader(
                  colLabel, colIndex, colsWithNonZeroCounts))}
          <th class=${totalColumnClasses}><div>Total</div></th>
        </tr>
        ${this.rowLabels.map((rowLabel, rowIndex) => this.renderRow(
            rowLabel, rowIndex, rowsWithNonZeroCounts, colsWithNonZeroCounts))}
        ${this.renderTotalRow(colsWithNonZeroCounts)}
      </table>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'data-matrix': DataMatrix;
  }
}
