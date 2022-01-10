/**
 * @fileoverview Element for controlling the faceting behavior of a module.
 *
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

// tslint:disable:no-new-decorators
import {customElement, property} from 'lit/decorators';
import {styleMap, StyleInfo} from 'lit/directives/style-map';
import {html} from 'lit';
import {observable} from 'mobx';

import {app} from '../core/app';
import {ReactiveElement} from '../lib/elements';
import {FacetingConfig, FacetingMethod, GroupService, NumericFeatureBins} from '../services/group_service';

import {styles as sharedStyles} from '../lib/shared_styles.css';
import {styles} from './faceting_control.css';

const ELEMENT_VISIBLE: StyleInfo = {
  display: 'block',
  visibility: 'visible'
};

const ELEMENT_HIDDEN: StyleInfo = {
  display: 'none',
  visibility: 'hidden'
};

/** The features and bins that should be used for faceting the dataset. */
export interface FacetsChange {
  features: string[];
  bins: NumericFeatureBins;
}

/** Controls for defining faceting behavior. */
@customElement('faceting-control')
export class FacetingControl extends ReactiveElement {
  private readonly groupService = app.getService(GroupService);

  @observable private expanded = false;
  @observable private features: string[] = [];
  @observable private bins: NumericFeatureBins = {};

  @observable @property({type: String}) contextName?: string;

  static override get styles() {
    return [sharedStyles, styles];
  }

  private toggleExpanded() {
    this.expanded = !this.expanded;
  }

  protected clickToClose(event: MouseEvent) {
    const path = event.composedPath();
    if (!path.some(elem => elem instanceof FacetingControl)) {
      this.expanded = false;
    }
  }

  override firstUpdated() {
    const onBodyClick = (event: MouseEvent) => {this.clickToClose(event);};
    this.reactImmediately(() => this.expanded, () => {
      if (this.expanded) {
        document.body.addEventListener(
          'click', onBodyClick, {passive: true, capture: true});
      } else {
        document.body.removeEventListener(
          'click', onBodyClick, {capture: true});
      }
    });
  }

  // TODO(b/204850097): Add UI supporting extended numeric configuration options
  renderFeatureOption (feature: string) {
    const checked = this.features.indexOf(feature) !== -1;
    const change = () => {
      const activeFacets = [...this.features];

      if (activeFacets.indexOf(feature) !== -1) {
        activeFacets.splice(activeFacets.indexOf(feature), 1);
      } else {
        activeFacets.push(feature);
      }

      const configs: FacetingConfig[] = activeFacets.sort()
          .filter(f => this.groupService.numericalFeatureNames.includes(f))
          .map(f => ({featureName: f, method: FacetingMethod.EQUAL_INTERVAL }));

      this.features = activeFacets;
      this.bins = this.groupService.numericalFeatureBins(configs);

      this.dispatchEvent(new CustomEvent<FacetsChange>('facets-change', {
        detail: {features: this.features, bins: this.bins}
      }));
    };

    return html`
      <div class="feature-options-row">
        <lit-checkbox ?checked=${checked} @change=${change} label=${feature}>
        </lit-checkbox>
      </div>`;
  }

  renderFeatureOptions() {
    return this.groupService.denseFeatureNames.map(
        feature => this.renderFeatureOption(feature));
  }

  override render() {
    const configPanelStyles = styleMap(this.expanded ? ELEMENT_VISIBLE :
                                                       ELEMENT_HIDDEN);

    const facetsList = this.features.length ?
      `${this.features.join(', ')} (${
         this.groupService.numIntersectionsLabel(this.bins, this.features)})` :
      'None';

    const forContext = this.contextName ? ` for ${this.contextName}` : '';

    const closeButtonClick = () => {this.expanded = false;};

    return html`
      <div class="faceting-info">
        <button class="hairline-button" @click=${this.toggleExpanded}
                title="Show or hide the faceting configuration${forContext}">
          <span class="material-icon">dashboard</span>
          Facets
        </button>
        <span class="active-facets">: ${facetsList}</span>
        <div class="config-panel popup-container" style=${configPanelStyles}>
          <div class="panel-header">
            <span class="panel-label">Faceting Config${forContext}</span>
            <mwc-icon class="icon-button min-button" @click=${closeButtonClick}>
              close
            </mwc-icon>
          </div>
          ${this.renderFeatureOptions()}
        </div>
      </div>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'faceting-control': FacetingControl;
  }
}
