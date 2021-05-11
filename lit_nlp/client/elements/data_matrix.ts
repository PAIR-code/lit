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

import * as d3 from 'd3';
import '@material/mwc-icon-button-toggle';
// tslint:disable:no-new-decorators
import {customElement, html, LitElement, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';
import {styles} from './data_matrix.css';


/**
 * Stores information for each confusion matrix cell.
 */
export interface MatrixCell {
  'ids': string[];
  'selected': boolean;
}

/**
 * An element that displays a confusion-matrix style vis for datapoints.
 */
@customElement('data-matrix')
export class DataMatrix extends LitElement {
  static get styles() {
    return [styles];
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
    for (let rowIndex = 0; rowIndex < this.matrixCells.length; rowIndex++) {
      const row = this.matrixCells[rowIndex];
      for (let colIndex = 0; colIndex < row.length; colIndex++) {
        const cell = row[colIndex];
        totalIds += cell.ids.length;
      }
    }
    return totalIds;
  }

  @computed
  get colorScale() {
    return d3.scaleLinear()
    .domain([0, this.totalIds])
    // Need to cast to numbers due to d3 typing.
    .range(["#F5F5F5" as unknown as number, "#006064" as unknown as number]);
  }

  private updateSelection() {
    let ids: string[] = [];
    for (const cellInfo of this.matrixCells.flat()) {
      if (cellInfo.selected) {
        ids = ids.concat(cellInfo.ids);
      }
    }
    const event = new CustomEvent('matrix-selection', {
      detail: {
        ids,
      }
    });
    this.dispatchEvent(event);
  }

  private renderColHeader(label: string, colIndex: number,
                          colsWithNonZeroCounts: Set<string>) {
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
  private renderCell(rowIndex: number, colIndex: number,
                     colsWithNonZeroCounts: Set<string>) {
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
    const backgroundColor = this.colorScale(cellInfo.ids.length);
    const percentage = cellInfo.ids.length / this.totalIds * 100;
    const textColor = percentage > 50 ? 'white' : 'black';
    const border = cellInfo.selected ?
        '2px solid #12B5CB' : '2px solid transparent';
    const cellStyle = styleMap({
      background: `${backgroundColor}`,
      color: `${textColor}`,
      border
    });
    return html`
        <td class="cell" style=${cellStyle} @click=${onCellClick}>
          <div class="cell-container">
            <div class="percentage">${percentage.toFixed(1)}%</div>
            <div class="val">(${cellInfo.ids.length})</div>
          </div>
        </td>`;
  }

  // Render a row of the confusion matrix, starting with the clickable
  // row header.
  private renderRow(rowLabel: string, rowIndex: number,
                    rowsWithNonZeroCounts: Set<string>,
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
      totalRowIds += cell.ids.length;
    }
    // clang-format off
    return html`
      <tr>
        ${rowIndex === 0 ? html`
            <td class='axis-title label-vertical'
                rowspan=${this.rowLabels.length + 1}>
              <div>${this.rowTitle}</div>
            </td>`
          : null}
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
      totalColIds += row[colIndex].ids.length;
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
      <mwc-icon-button-toggle class="icon-button"
        title="Rotate column labels"
        onIcon="text_rotate_up" offIcon="text_rotation_none"
        ?on="${this.verticalColumnLabels}"
        @MDCIconButtonToggle:change="${toggleVerticalColumnLabels}"
        @icon-button-toggle-change="${toggleVerticalColumnLabels}">
      </mwc-icon-button-toggle>
    `;
    // clang-format on
  }

  render() {
    if (this.matrixCells.length === 0) {
      return null;
    }

    const rowsWithNonZeroCounts = new Set<string>();
    const colsWithNonZeroCounts = new Set<string>();
    for (let rowIndex = 0; rowIndex < this.matrixCells.length; rowIndex++) {
      const row = this.matrixCells[rowIndex];
      for (let colIndex = 0; colIndex < row.length; colIndex++) {
        const cell = row[colIndex];
        if (cell.ids.length > 0) {
          rowsWithNonZeroCounts.add(this.rowLabels[rowIndex]);
          colsWithNonZeroCounts.add(this.colLabels[colIndex]);
        }
      }
    }

    const totalColumnClasses = classMap({
        'total-title-cell': true,
        'align-bottom': this.verticalColumnLabels
      });

    // clang-format off
    return html`
      <table>
        <tr>
          <th>${this.renderColumnRotateButton()}</th><td></td>
          <td class='axis-title' colspan=${this.colLabels.length}>
            ${this.colTitle}
          </td>
        </tr>
        <tr>
          <td colspan=2></td>
          ${this.colLabels.map(
              (colLabel, colIndex) => this.renderColHeader(
                  colLabel, colIndex, colsWithNonZeroCounts))}
          <th class=${totalColumnClasses}><div>Total</div></th>
        </tr>
        ${this.rowLabels.map(
            (rowLabel, rowIndex) => this.renderRow(
                rowLabel, rowIndex, rowsWithNonZeroCounts,
                colsWithNonZeroCounts))}
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
