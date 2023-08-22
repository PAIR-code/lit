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
import {computed, observable} from 'mobx';

import {CATEGORICAL_NORMAL, DEFAULT, MULTIHUE_CONTINUOUS, SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap} from '../lib/colors';
import {ColorOption, D3Scale, IndexedInput} from '../lib/types';

import {DataService} from './data_service';
import {GroupService} from './group_service';
import {LitService} from './lit_service';
import {ColorObservedByUrlService, UrlConfiguration} from './url_service';

export {SalienceCmap, SignedSalienceCmap, UnsignedSalienceCmap};

/**
 * A singleton class that handles all coloring options.
 */
export class ColorService extends LitService implements
ColorObservedByUrlService {
  constructor(
      private readonly groupService: GroupService,
      private readonly dataService: DataService) {
    super();
  }

  private readonly defaultColor = DEFAULT;

  private readonly defaultOption: ColorOption = {
    name: 'None',
    getValue: (input: IndexedInput) => 'all',
    scale: d3.scaleOrdinal([this.defaultColor]).domain(['all']) as D3Scale,
  };

  // Name of selected feature to color datapoints by, or default not coloring by
  // features.
  @observable mySelectedColorOption = this.defaultOption;
  // It's used for the url service. When urlService.syncStateToUrl is invoked,
  // colorableOptions are not available. There, this variable is used to
  // preserve the url param value entered by users.
  @observable
  selectedColorOptionName: string = '';

  // All variables that affect color settings, so clients can listen for when
  // they may need to rerender.
  @computed get all() {
    return [
      this.selectedColorOption,
    ];
  }

  // Return the selectedColorOption based on the selectedColorOptionName
  @computed get selectedColorOption() {
    if (this.colorableOptions.length === 0) {
      return this.defaultOption;
    }
    const selectedOption = this.getTargetColorOption(
        this.selectedColorOptionName, this.colorableOptions);
    if (selectedOption !== undefined) {
      return selectedOption;
    }
    const defaultClassificationOption = this.getTargetColorOption(
        this.defaultClassificationColorOption, this.colorableOptions);
    if (defaultClassificationOption !== undefined) {
      return defaultClassificationOption;
    }
    return this.defaultOption;
  }

  @computed get colorableOptions() {
    // TODO(b/156100081): Get proper reactions on data service columns.
    // tslint:disable:no-unused-variable Causes recompute on change.
    const data = this.dataService.dataVals;
    const catInputFeatureOptions =
        this.groupService.categoricalFeatureNames.map((feature: string) => {
          const domain = this.groupService.categoricalFeatures[feature];
          let range = this.dataService.getColumnInfo(feature)?.colorRange;
          if (range == null) {
            range = domain.length > 1 ? CATEGORICAL_NORMAL : [DEFAULT];
          }
          return {
            name: feature,
            getValue: (input: IndexedInput) =>
                this.dataService.getVal(input.id, feature),
            scale: d3.scaleOrdinal(range as string[]).domain(domain) as D3Scale
          };
        });
    const boolInputFeatureOptions =
        this.groupService.booleanFeatureNames.map((feature: string) => {
          const domain = ['false', 'true'];
          let range = this.dataService.getColumnInfo(feature)?.colorRange;
          if (range == null) {
            range = CATEGORICAL_NORMAL;
          }
          return {
            name: feature,
            getValue: (input: IndexedInput) =>
                this.dataService.getVal(input.id, feature),
            scale: d3.scaleOrdinal(range as string[]).domain(domain) as D3Scale
          };
        });
    const numInputFeatureOptions =
        this.groupService.numericalFeatureNames.map((feature: string) => {
          const domain = this.groupService.numericalFeatureRanges[feature];
          let range = this.dataService.getColumnInfo(feature)?.colorRange;
          if (range == null) {
            range = MULTIHUE_CONTINUOUS;
          }
          return {
            name: feature,
            getValue: (input: IndexedInput) =>
                this.dataService.getVal(input.id, feature),
            scale: d3.scaleSequential(range as (t: number) => string)
                       .domain(domain) as D3Scale
          };
        });
    return [
      ...catInputFeatureOptions, ...numInputFeatureOptions,
      ...boolInputFeatureOptions, this.defaultOption
    ];
  }

  @computed
  get defaultClassificationColorOption() {
    return this.dataService.predictedClassFeatureName;
  }

  // Return the target color option if its name exists in the available options.
  // Otherwise, if the target color option name does not exist in the available
  // options, return undefined.
  getTargetColorOption(
      targetColorOptionName: string,
      availableColorOptions: ColorOption[]): ColorOption|undefined {
    return availableColorOptions.find(
        option => option.name === targetColorOptionName);
  }

  // Return the color for the provided datapoint.
  getDatapointColor(input?: IndexedInput|null) {
    if (this.selectedColorOption == null || input == null) {
      return this.defaultColor;
    }
    const val = this.selectedColorOption.getValue(input);
    return this.selectedColorOption.scale(val) || this.defaultColor;
  }

  // Set color option based on the URL configuration
  setUrlConfiguration(urlConfiguration: UrlConfiguration) {
    this.selectedColorOptionName = urlConfiguration.colorBy ?? '';
  }
}
