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
import {DEFAULT, CATEGORICAL_NORMAL, CONTINUOUS_UNSIGNED_LAB, CONTINUOUS_SIGNED_LAB, MULTIHUE_CONTINUOUS} from '../lib/colors';

import {ClassificationService} from './classification_service';
import {GroupService} from './group_service';
import {LitService} from './lit_service';
import {RegressionService} from './regression_service';
import {AppState} from './state_service';

/** Color map for salience maps. */
export abstract class SalienceCmap {
  /**
   * An RGB interpolated color scale for one of the continuous LAB ramps from
   * VizColor, which have been linearized.
   */
  protected colorScale: d3.ScaleSequential<string>;

  // Exponent for computing luminance values from salience scores.
  // A higher value gives higher contrast for small (close to 0) salience
  // scores.
  // See https://en.wikipedia.org/wiki/Gamma_correction
  constructor(protected gamma: number = 1.0,
              protected domain: [number, number] = [0, 1]) {
    this.colorScale = d3.scaleSequential(CONTINUOUS_UNSIGNED_LAB).domain(domain);
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
    const [min, max] = this.colorScale.domain();
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
    return this.colorScale(this.lightness(d));
  }
}

/** Color map for signed salience maps. */
export class SignedSalienceCmap extends SalienceCmap {
  constructor(gamma: number = 1.0, domain: [number, number] = [-1, 1]) {
    super(gamma, domain);
    this.colorScale = d3.scaleSequential(CONTINUOUS_SIGNED_LAB).domain(domain);
  }

  bgCmap(d: number): string {
    const direction = d < 0 ? -1 : 1;
    return this.colorScale(this.lightness(d) * direction);
  }
}

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
    reaction(() => this.appState.currentModels, currentModels => {
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
