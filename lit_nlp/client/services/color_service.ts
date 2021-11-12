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

// tslint:disable:no-new-decorators
import * as d3 from 'd3';
import {computed, observable, reaction} from 'mobx';

import {ColorOption, D3Scale, IndexedInput} from '../lib/types';
import {DEFAULT, CATEGORICAL_NORMAL, MULTIHUE_CONTINUOUS} from '../lib/colors';

import {ClassificationService} from './classification_service';
import {GroupService} from './group_service';
import {LitService} from './lit_service';
import {RegressionService} from './regression_service';
import {AppState} from './state_service';

/**
 * A singleton class that handles all coloring options.
 */
export class ColorService extends LitService {
  constructor(
      private readonly appState: AppState,
      private readonly groupService: GroupService,
      private readonly classificationService: ClassificationService,
      private readonly regressionService: RegressionService) {
    super();
    reaction(() => appState.currentModels, currentModels => {
      this.reset();
    });
  }

  private readonly defaultColor = DEFAULT;

  private readonly defaultOption: ColorOption = {
    name: 'None',
    getValue: (input: IndexedInput) => 'all',
    scale: d3.scaleOrdinal([this.defaultColor]).domain(['all']) as D3Scale,
  };

  // Name of selected feature to color datapoints by, or default not coloring by
  // features.
  @observable selectedColorOption = this.defaultOption;

  // All variables that affect color settings, so clients can listen for when
  // they may need to rerender.
  @computed
  get all() {
    return [
      this.selectedColorOption,
      this.classificationService.allMarginSettings
    ];
  }

  @computed
  get colorableOptions() {
    const catInputFeatureOptions =
        this.groupService.categoricalFeatureNames.map((feature: string) => {
          const domain = this.groupService.categoricalFeatures[feature];
          const range = domain.length > 1 ? CATEGORICAL_NORMAL : [ DEFAULT ];
          return {
            name: feature,
            getValue: (input: IndexedInput) => input.data[feature],
            scale: d3.scaleOrdinal(range).domain(domain) as D3Scale
          };
        });
    const numInputFeatureOptions =
        this.groupService.numericalFeatureNames.map((feature: string) => {
          const domain = this.groupService.numericalFeatureRanges[feature];
          return {
            name: feature,
            getValue: (input: IndexedInput) => input.data[feature],
            scale: d3.scaleSequential(MULTIHUE_CONTINUOUS)
                     .domain(domain) as D3Scale
          };
        });
    return [
      ...catInputFeatureOptions, ...numInputFeatureOptions,
      ...this.classificationService.colorOptions,
      ...this.regressionService.colorOptions, this.defaultOption
    ];
  }

  // Return the color for the provided datapoint.
  getDatapointColor(input?: IndexedInput|null) {
    if (this.selectedColorOption == null || input == null) {
      return this.defaultColor;
    }
    const val = this.selectedColorOption.getValue(input);
    return this.selectedColorOption.scale(val) || this.defaultColor;
  }

  /**
   * Reset stored info. Used when active models change.
   */
  reset() {
    this.selectedColorOption = this.defaultOption;
  }
}
