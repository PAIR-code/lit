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

import '@material/mwc-icon-button-toggle';
// tslint:disable:no-new-decorators
import {customElement, html, LitElement, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {observable} from 'mobx';
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

    // Render a clickable column header cell.
    const renderColHeader = (label: string, colIndex: number) => {
      const onColClick = () => {
        const cells = this.matrixCells.map((cells) => cells[colIndex]);
        const allSelected = cells.every((cell) => cell.selected);
        cells.forEach((cell) => {
          cell.selected = !allSelected;
        });
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
    };

    // Render a clickable confusion matrix cell.
    const renderCell = (rowIndex: number, colIndex: number) => {
      if (this.matrixCells[rowIndex]?.[colIndex] == null) {
        return null;
      }
      const cellInfo = this.matrixCells[rowIndex][colIndex];
      const cellClasses = classMap({
        cell: true,
        selected: cellInfo.selected,
        diagonal: colIndex === rowIndex,
      });
      const onCellClick = () => {
        cellInfo.selected = !cellInfo.selected;
        this.updateSelection();
        this.requestUpdate();
      };
      if (this.hideEmptyLabels &&
          !colsWithNonZeroCounts.has(this.colLabels[colIndex])) {
        return null;
      }
      return html`
          <td class=${cellClasses} @click=${onCellClick}>
            ${cellInfo.ids.length}
          </td>`;
    };

    // Render a row of the confusion matrix, starting with the clickable
    // row header.
    const renderRow = (rowLabel: string, rowIndex: number) => {
      const onRowClick = () => {
        const cells = this.matrixCells[rowIndex];
        const allSelected = cells.every((cell) => cell.selected);
        cells.forEach((cell) => {
          cell.selected = !allSelected;
        });
        this.updateSelection();
        this.requestUpdate();
      };
      if (this.hideEmptyLabels && !rowsWithNonZeroCounts.has(rowLabel)) {
        return null;
      }
      // clang-format off
      return html`
        <tr>
          ${rowIndex === 0 ? html`
              <td class='axis-title label-vertical' rowspan=${this.rowLabels.length}>
                <div>${this.rowTitle}</div>
              </td>`
            : null}
          <th class="header-cell align-right" @click=${onRowClick}>
            ${rowLabel}
          </th>
          ${this.colLabels.map(
              (colLabel, colIndex) => renderCell(rowIndex, colIndex))}
        </tr>`;
      // clang-format on
    };

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
              (colLabel, colIndex) => renderColHeader(colLabel, colIndex))}
        </tr>
        ${this.rowLabels.map(
            (rowLabel, rowIndex) => renderRow(rowLabel, rowIndex))}
      </table>
    `;
    // clang-format on
  }

  renderColumnRotateButton() {
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
}

declare global {
  interface HTMLElementTagNameMap {
    'data-matrix': DataMatrix;
  }
}
