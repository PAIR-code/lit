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
 * Custom documentation module
 */

// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators';
import {css, html} from 'lit';
import {computed} from 'mobx';
import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {ModelInfoMap, Spec} from '../lib/types';
import {getTemplateStringFromMarkdown} from '../lib/utils';
import {AppState} from '../services/services';
import {styles as sharedStyles} from '../lib/shared_styles.css';


/**
 * Module for showing client-provided documentation.
 */
@customElement('documentation-module')
export class DocumentationModule extends LitModule {
  static override get styles() {
    const styles = css`
      .wrapper {
        margin: 8px;
      }
    `;
    return [sharedStyles, styles];
  }

  static override title = 'Documentation';
  static override numCols = 3;
  static override duplicateForModelComparison = false;

  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => html`
      <documentation-module model=${model} .shouldReact=${shouldReact}
        selectionServiceIndex=${selectionServiceIndex}>
      </documentation-module>`;

  @computed
  get markdownString() {
    return getTemplateStringFromMarkdown(this.appState.metadata.inlineDoc!);
  }

  override renderImpl() {
    return html`<div class="wrapper">${this.markdownString}</div>`;
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap,
                                      datasetSpec: Spec) {
    // Only show if inline documentation provided in metadata.
    const appState = app.getService(AppState);
    return appState.metadata?.inlineDoc != null;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'documentation-module': DocumentationModule;
  }
}
