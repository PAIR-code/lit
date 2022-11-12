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
 * LIT documentation splash screen
 */

// tslint:disable:no-new-decorators
import '../elements/checkbox';

import {html} from 'lit';
import {customElement, query} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {computed, observable} from 'mobx';

import {ReactiveElement} from '../lib/elements';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {getTemplateStringFromMarkdown} from '../lib/utils';
import {AppState} from '../services/services';

import {app} from './app';
import {styles} from './documentation.css';

/**
 * The documentation page.
 */
@customElement('lit-documentation')
export class DocumentationComponent extends ReactiveElement {
  @observable isOpen = false;
  @observable currentPage = 0;

  static override get styles() {
    return [sharedStyles, styles];
  }

  // Markdown of splash screen text to render.
  @observable
  private readonly markdownPages = [
    `![](static/onboarding_1_welcome.gif)\n# Welcome to LIT!\nLIT is a ` +
        `visual, interactive tool to help ML researchers, engineers, product ` +
        `teams, and decision makers explore and compare models using ` +
        `out-of-the-box interpretability methods to improve performance and ` +
        `mitigate bias issues. We support a variety of models, data types, and ` +
        `interpretability methods.`,

    `![](static/onboarding_2_modules.gif)\n# Start interacting with flexible ` +
        `modules\nLIT uses modules that allow rich interactions using ` +
        `visualizations, tables, to perform lots of tasks. Modules interact with ` +
        `each other so you can seamlessly switch between various workflows – ` +
        `from datapoint explorations, to model predictions to counterfactuals – ` +
        `without getting lost.`,

    `![](static/onboarding_3_workspaces.gif)\n# Workspaces in LIT\nWhether ` +
        `you're evaluating model predictions, looking for explanations, or ` +
        `generating counterfactuals, you can find the modules you need in ` +
        `LIT's workflow-optimized workspaces.\n`,

    `![](static/onboarding_4_explore.gif)\n# Explore data and models\n` +
        `Explore model performance across the entire dataset, a subset, or ` +
        `individual data points. Create subsets of your loaded data by filtering ` +
        `on criteria, curating by selection, or by generating “what-if” ` +
        `style counterfactuals.`,

    `![](static/onboarding_5_toolbars.gif)\n# Manage models, datasets, and ` +
        `selections\nLIT's toolbars present controls for selecting models and ` +
        `datasets and sharing findings. Advanced controls allow you to navigate ` +
        `individual data points systematically or switch into datapoint ` +
        `comparison mode.`,

    `![](static/onboarding_6_compare.gif)\n# Dynamically evaluate models ` +
        `side-by-side\nEasily switch between exploring a single model or ` +
        `comparing multiple models from within your workflow. Compare ` +
        `predictions, explanations, and metrics to see differences in behavior ` +
        `and performance.`,

    `![](static/onboarding_7_start.gif)\n# Start exploring!\nNeed more help? ` +
        `Visit our documentation page for more advanced help and support. ` +
        `Otherwise, click get started.`
  ];

  @query('#holder') docElement!: HTMLDivElement;

  constructor(private readonly appState = app.getService(AppState)) {
    super();
  }

  override firstUpdated() {
    // Sync if this is open or closed with appState to control URL param setting
    // when splash screen is open.
    const isOpen = () => this.isOpen;
    this.react(isOpen, () => {
      this.appState.documentationOpen = this.isOpen;
    });

    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        this.close();
      } else if (e.key === 'ArrowLeft' && this.currentPage > 0) {
        this.currentPage -= 1;
      } else if (e.key === 'ArrowRight') {
        if (this.currentPage < this.totalPages - 1) {
          this.currentPage += 1;
        } else {
          this.close();
        }
      }
    });

    // The splash screen will be open if the URL param explictly states to show
    // it on load.
    this.isOpen = this.appState.documentationOpen;
  }

  /**
   * Open the documentation dialog.
   */
  open() {
    this.currentPage = 0;
    this.isOpen = true;
  }

  /**
   * Close the documentation dialog.
   */
  close() {
    this.isOpen = false;
  }

  /**
   * Get template strings to render based on provided doc from metadata and
   * our built-in documentation.
   */
  @computed
  get pagesToRender() {
    const pages = this.markdownPages;
    if (this.appState.metadata.onboardStartDoc != null) {
      pages.unshift(this.appState.metadata.onboardStartDoc);
    }
    if (this.appState.metadata.onboardEndDoc != null) {
      pages.splice(pages.length - 1, 0, this.appState.metadata.onboardEndDoc);
    }
    return pages.map(page => getTemplateStringFromMarkdown(page));
  }

  @computed
  get totalPages() {
    return this.pagesToRender.length;
  }

  override render() {
    const hiddenClassMap = classMap({hide: !this.isOpen});
    const docToDisplay = this.pagesToRender[this.currentPage];
    const onCloseClick = () => {
      this.isOpen = false;
    };

    // clang-format off
    return html`
      <div id="doc-holder">
        <div id="overlay" class=${hiddenClassMap}
         @click=${() => { this.close(); }}></div>
        <div id="doc" class=${hiddenClassMap}>
          <div class="top-section">
            <div id="holder">
               ${docToDisplay}
            </div>
          </div>
          <div class="bottom-area">
            <div class="dots-line">
              <div class="dots-holder">
                ${this.pagesToRender.map((page, i) => this.renderDot(i))}
              </div>
              <mwc-icon class="icon-button close-button" @click=${onCloseClick}>
                close
              </mwc-icon>
            </div>
            <div class="page-controls">
              <div class="doc-link-holder">
                ${this.renderDocLink()}
              </div>
              <div class="prev-next-container">
                ${this.renderNextPrevButton(false)}
                ${this.renderNextPrevButton(true)}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    // clang-format on
  }

  private renderDocLink() {
    const help = 'https://github.com/PAIR-code/lit/wiki';
    return html`
      <div>
        <a href="${help}" target="_blank">
          Documentation
          <mwc-icon class="doc-open-icon">open_in_new</mwc-icon>
        </a>
      </div>
    `;
  }

  private renderDot(idx: number) {
    const onClick = () => {
      this.currentPage = idx;
    };

    const classes = classMap({
      'dot': true,
      'filled': this.currentPage === idx,
      'clickable': this.currentPage !== idx,
    });
    return html`<div class=${classes} @click=${onClick}></div>`;
  }

  private renderNextPrevButton(next = true) {
    const icon = next ? 'east' : 'west';  // Arrow direction.
    const lastPage = this.currentPage === this.totalPages - 1;
    const onClick = () => {
      if (lastPage && next) {
        this.isOpen = false;
        return;
      }
      if (next) {
        this.currentPage++;
      } else {
        this.currentPage--;
      }
    };
    const text = next ? (lastPage ? 'Get started' : 'Next') : 'Back';
    const disabled = !next && this.currentPage === 0;
    // clang-format off
    const renderNextIcon = () => {
      if (lastPage) {
        return null;
      }
      return html`<mwc-icon class='nav-icon'>${icon}</mwc-icon>`;
    };
    return html`
     <button class='hairline-button nav-button' @click=${onClick} ?disabled=${disabled}>
       ${next ? null : html`<mwc-icon class='nav-icon'>${icon}</mwc-icon>`}
       ${text}
       ${next ? renderNextIcon() : null}
    </button>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-documentation': DocumentationComponent;
  }
}
