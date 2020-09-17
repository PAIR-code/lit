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

  private readonly defaultColor = d3.schemeCategory10[0];

  private readonly defaultOption: ColorOption = {
    name: 'none',
    getValue: (input: IndexedInput) => 'all',
    scale: d3.scaleOrdinal([this.defaultColor]).domain(['all']) as D3Scale,
  };

  // Name of selected feature to color datapoints by, or default not coloring by
  // features.
  @observable selectedColorOption = this.defaultOption;

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
          return {
            name: feature,
            getValue: (input: IndexedInput) => input.data[feature],
            scale:
                d3.scaleOrdinal(d3.schemeCategory10)
                    .domain(this.groupService.categoricalFeatures[feature]) as
                D3Scale
          };
        });
    const numInputFeatureOptions =
        this.groupService.numericalFeatureNames.map((feature: string) => {
          return {
            name: feature,
            getValue: (input: IndexedInput) => input.data[feature],
            scale: d3.scaleSequential(d3.interpolateViridis)
                       .domain(
                           this.groupService.numericalFeatureRanges[feature]) as
                D3Scale
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
    const color = this.selectedColorOption.scale(val);
    if (color == null) {
      return this.defaultColor;
    }
    return color;
  }

  /**
   * Reset stored info. Used when active models change.
   */
  reset() {
    this.selectedColorOption = this.defaultOption;
  }
}
