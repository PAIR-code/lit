/**
 * @fileoverview A resusable component for downloading data in LIT
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

// tslint:disable:no-new-decorators
import './popup_container';

import {html} from 'lit';
import {classMap} from 'lit/directives/class-map.js';
import {customElement, property} from 'lit/decorators.js';
import * as papa from 'papaparse';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';

import {styles} from './export_controls.css';
import {PopupContainer} from './popup_container';
import {SortableTableEntry} from './table';

/**
 * An element that handles logic for downloading data.
 */
@customElement('export-controls')
export class ExportControls extends ReactiveElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  /** The default file download name. */
  @property({type: String}) downloadFilename = 'data.csv';
  /** A list of rows of data to download. */
  @property({type: Object}) data: SortableTableEntry[][] = [];
  /** Column names. */
  @property({type: Object}) columnNames: string[] = [];
  /** Download popup position defaults to below the download icon. */
  @property({type: String}) popupPosition = 'below';
  @property({type: String}) tooltipPosition = 'left';
  /** If true, disable controls. */
  @property({type: Boolean}) disabled = false;

  getCSVContent(): string {
    return papa.unparse(
        {fields: this.columnNames, data: this.data},
        {newline: '\r\n'});
  }

  getPopupClasses() {
    return classMap({
      'hidden': this.disabled,
      'download-popup': true,
      'above': this.popupPosition === 'above'
    });
  }

  /**
   * Renders the copy and download buttons to download data.
   */
  override render() {
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
      this.downloadFilename = (e.target as HTMLInputElement).value;
    };

    function onEnter(e: KeyboardEvent) {
      if (e.key === 'Enter') downloadCSV();
    }

    const iconClass = classMap({
      'icon-button': true,
      'mdi-outlined': true,
      'disabled': this.disabled,
    });

    // TODO(b/265978596): Consider a better place for download tooltips.
    // clang-format off
    return html`
      <div id='export-controls'>
        <lit-tooltip content="Copy ${this.data.length} rows as CSV"
          tooltipPosition=${this.tooltipPosition}>
          <mwc-icon class=${iconClass} slot="tooltip-anchor" @click=${copyCSV}>
            file_copy
          </mwc-icon>
        </lit-tooltip>

        <popup-container class='${this.getPopupClasses()}'>
          <lit-tooltip content="Download ${this.data.length} rows as CSV"
            tooltipPosition="${this.tooltipPosition}" slot="toggle-anchor">
            <mwc-icon class=${iconClass} slot="tooltip-anchor">
              file_download
            </mwc-icon>
          </lit-tooltip>
          <div class='download-popup-controls'>
            <label for="filename">Filename</label>
            <input type="text" name="filename" value=${this.downloadFilename}
              @input=${updateFilename} @keydown=${onEnter}>
            <button class='download-button filled-button nowrap'
              @click=${downloadCSV}
              ?disabled=${!this.downloadFilename}>
              Download ${this.data.length} rows
            </button>
          </div>
        </popup-container>
      </div>`;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'export-controls': ExportControls;
  }
}
