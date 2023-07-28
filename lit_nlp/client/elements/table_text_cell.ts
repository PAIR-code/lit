/**
 * @fileoverview A reusable element for expandable text cells in LIT tables
 *
 * @license
 * Copyright 2023 Google LLC
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
import {css, html} from 'lit';
import {customElement, property} from 'lit/decorators.js';
import {styleMap} from 'lit/directives/style-map.js';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay} from '../lib/types';

const PX_TO_CHAR = 0.139;
const SHOW_MORE_WIDTH_PX = 25;
const LINES_TO_RENDER = 3;
const DEFAULT_MAX_WIDTH = 200;

/**
 * A text cell element that expands hidden text on click.
 *
 *
 * Usage:
 *   <lit-table-text-cell
 *      .content=${content}
 *      .maxWidth=${maxWidth}
 *      @showmore=${showMore}>
 *   </lit-table-text-cell>
 */
@customElement('lit-table-text-cell')
export class LitTableTextCell extends ReactiveElement {
    static override get styles() {
      return [
        sharedStyles, css`
        .lit-table-text-cell {
          max-height: 150px;
          overflow: auto;
          white-space: pre-wrap;
          display: inline;
        }

        lit-showmore {
          white-space: normal;
        }
      `
      ];
    }

    @property({type: String}) content = '';
    @property({type: Number}) maxWidth?: number;
    @property({type: Boolean}) expanded = false;

    /**
     * Renders the Show More element with hidden or visible test.
     */
    override render() {
      const onClickShowMore = (e:Event) => {
        this.expanded = true;
        const event = new CustomEvent('showmore');
        this.dispatchEvent(event);
        e.stopPropagation();
        e.preventDefault();
      };

      const maxWidth = this.maxWidth ?? DEFAULT_MAX_WIDTH;

      const showMoreStyles = styleMap({
        '--show-more-max-width': `${maxWidth}px`,
        '--show-more-icon-width': `20px`,
      });

      const clippedTextLengthPx =
        maxWidth*LINES_TO_RENDER - SHOW_MORE_WIDTH_PX;

      const clippedByMaxWidthChars = Math.floor(clippedTextLengthPx*PX_TO_CHAR);

      const clippedByLineBreaksChars =
        this.content.split("\n\n", LINES_TO_RENDER).join("\n\n").length;

      const clippedTextLengthChars =
        Math.min(clippedByMaxWidthChars, clippedByLineBreaksChars);

      const isExpanded =
        this.expanded ||  this.content.length <= clippedTextLengthChars;

      const hiddenChars = this.content.length - clippedTextLengthChars;

      const visibleText = isExpanded ?
          this.content :
          this.content.substring(0, clippedTextLengthChars);

      const displayText = formatForDisplay(visibleText, undefined, true);

      // div.lit-table-text-cell uses pre-wrap, so HTML template must take
      // care to avoid introducing extraneous whitespace or newlines.
      // clang-format off
      const renderShowMore = isExpanded ? "" :
        html`<lit-showmore .hiddenTextLength=${hiddenChars}
              @showmore=${onClickShowMore}></lit-showmore>`;

      return html`
       <div style=${showMoreStyles}
        class='lit-table-text-cell'>${displayText} ${renderShowMore}</div>`;
      // clang-format on
    }
  }

  declare global {
    interface HTMLElementTagNameMap {
      'lit-table-text-cell': LitTableTextCell;
    }
  }
