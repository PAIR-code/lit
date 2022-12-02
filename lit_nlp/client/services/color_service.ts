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

import {CATEGORICAL_NORMAL, CONTINUOUS_SIGNED_LAB, CONTINUOUS_UNSIGNED_LAB, DEFAULT, MULTIHUE_CONTINUOUS} from '../lib/colors';
import {ColorOption, D3Scale, IndexedInput} from '../lib/types';

import {DataService} from './data_service';
import {GroupService} from './group_service';
import {LitService} from './lit_service';
import {ColorObservedByUrlService, UrlConfiguration} from './url_service';

/** Color map for salience maps. */
export abstract class SalienceCmap {
  /**
   * An RGB interpolated color scale for one of the continuous LAB ramps from
   * VizColor, which have been linearized.
   */
  protected myColorScale: d3.ScaleSequential<string>;

  get colorScale() { return this.myColorScale; }

  // Exponent for computing luminance values from salience scores.
  // A higher value gives higher contrast for small (close to 0) salience
  // scores.
  // See https://en.wikipedia.org/wiki/Gamma_correction
  constructor(protected gamma: number = 1.0,
              protected domain: [number, number] = [0, 1]) {
    this.myColorScale = d3.scaleSequential(CONTINUOUS_UNSIGNED_LAB).domain(domain);
  }

  /**
   * Determine the correct text color -- black or white -- given the lightness
   * for this datum
   */
  textCmap(d: number): string {
    return (this.lightness(d) < 0.5) ? 'black' : 'white';
  }

  /** Clamps the value of d to the color scale's domain */
  clamp(d: number): number {
    const [min, max] = this.myColorScale.domain();
    return Math.max(min, Math.min(max, d));
  }

  /** Gamma corrected lightness in the range [0, 1]. */
  lightness(d: number): number {
    d = Math.abs(this.clamp(d));
    // Gamma correction to increase visibility of low salience datapoints
    d = (1 - d) ** this.gamma;
    // Invert direction because HSL and our color scales place white on opposite
    // ends of the [0, 1] range
    return 1 - d;
  }

  /** Color mapper. More extreme salience values get darker colors. */
  abstract bgCmap(d: number): string;
}

/** Color map for unsigned (positive) salience maps. */
export class UnsignedSalienceCmap extends SalienceCmap {
  bgCmap(d: number): string {
    return this.myColorScale(this.lightness(d));
  }
}

/** Color map for signed salience maps. */
export class SignedSalienceCmap extends SalienceCmap {
  constructor(gamma: number = 1.0, domain: [number, number] = [-1, 1]) {
    super(gamma, domain);
    this.myColorScale = d3.scaleSequential(CONTINUOUS_SIGNED_LAB).domain(domain);
  }

  bgCmap(d: number): string {
    const direction = d < 0 ? -1 : 1;
    return this.myColorScale(this.lightness(d) * direction);
  }
}

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
  @observable selectedColorOptionName: string = '';

  // All variables that affect color settings, so clients can listen for when
  // they may need to rerender.
  @computed
  get all() {
    return [
      this.selectedColorOption,
    ];
  }

  // Return the selectedColorOption based on the selectedColorOptionName
  @computed
  get selectedColorOption() {
    if (this.colorableOptions.length === 0 ||
        this.selectedColorOptionName.length === 0) {
      return this.defaultOption;
    } else {
      const options = this.colorableOptions.filter(
          option => option.name === this.selectedColorOptionName);
      return options.length ? options[0] : this.defaultOption;
    }
  }

  @computed
  get colorableOptions() {
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
