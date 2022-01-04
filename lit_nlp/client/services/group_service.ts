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
 * Group Service deals with the logic for categorical and numeric
 * grouping.
 */

// tslint:disable:no-new-decorators
import * as d3 from 'd3';  // Used for creating bins, not visualization.
import {computed} from 'mobx';

import {FacetMap, GroupedExamples, IndexedInput} from '../lib/types';
import {facetMapToDictKey, findSpecKeys, roundToDecimalPlaces} from '../lib/utils';

import {DataForInput, DataService} from './data_service';
import {LitService} from './lit_service';
import {AppState} from './state_service';

/**
 * A map of categorical features to their possible values.
 */
export interface CategoricalFeatures {
  [feature: string]: string[];
}

/**
 * A map of numeric features to their min and max values.
 */
export interface NumericFeatures {
  [feature: string]: [number, number];
}

interface NumericBins {
  [name: string]: number[];
}

/**
 * A map of numeric features to their categorical bins.
 */
export interface NumericFeatureBins {
  [feature: string]: NumericBins;
}

/**
 * Function type for getting a (possibly binned) feature value from an
 * IndexedInput. Inputs are the faceting configs, the datapoint, its index in
 * the current data, and the feature of interest.
 */
export type GetFeatureFunc = (
    bins: NumericFeatureBins, datum: IndexedInput, index: number,
    feature: string) => number | string | number[] | null;

/**
 * Enumeration of the different faceting methods supported by the GroupService.
 *
 * -  DISCRETE: Creates a bin for every discrete value for this feature. For
 *    numerical features, this is derived by using the step configured for that
 *    feature in the dataset spec to traverse the features range. For
 *    categorical features, it creates a bin for each vaue in the vocabulary.
 *    For binary features, it creates 2 bins.
 * -  EQUAL_INTERVAL: Divides the domain into N equal-width bins
 * -  QUANTILE: Divides the range into N bins with an equal number of datapoints
 * -  THRESHOLD: Divides the range into 2 bins, [m, t) and [t, M] where m is the
 *    minimum value of the range, M is the maximum value of the range, and t is
 *    the threshold
 */
export enum FacetingMethod {
  DISCRETE = 'discrete',
  EQUAL_INTERVAL = 'equal-interval',
  QUANTILE = 'quantile',
  THRESHOLD = 'threshold'
}

/**
 * Definition for how to facet a feature into a series of bins
 *
 * The feature type associated with featureName impacts the behavior as follows.
 *
 * *  Boolean features: method, numBins, and threshold are ignored as these
 *    features always produce 2 bins, one for `true` and one for `false`.
 * *  Categorical features: method, numBins, and threshold are ignored as these
 *    features always produce one bin for each of the labels in the vocabulary
 * *  Numerical features
 *     *  `method` is strongly encouraged, omission will result in the
 *        generation of DISCRETE bins.
 *     *  `numBins` is encouraged for the EQUAL_INTERVAL method; when provided
 *        and >= 1, the service generates that many bins, otherwise the service
 *        infers the correct number of bins to generate using the Freedman-
 *        Diaconis algorithm.
 *     *  `numBins` is required the QUANTILE method, where it must be >= 1.
 *     *  `threshold` is requires for the THRESHOLD method.
 */
export interface FacetingConfig {
  featureName: string;
  method?: FacetingMethod;
  numBins?: number;
  threshold?: number;
}

/**
 * A singleton class that handles grouping.
 */
export class GroupService extends LitService {
  constructor(private readonly appState: AppState,
              private readonly dataService: DataService) {
    super();
  }

  /** Get the names of categorical features. */
  @computed
  get categoricalFeatureNames(): string[] {
    const dataSpec = this.appState.currentDatasetSpec;
    const names = findSpecKeys(dataSpec, 'CategoryLabel');
    return names;
  }

  /** Get the names of the numerical features. */
  @computed
  get numericalFeatureNames(): string[] {
    return this.dataService.scalarCols;
  }

  /** Get the names of the boolean features. */
  @computed
  get booleanFeatureNames(): string[] {
    const dataSpec = this.appState.currentDatasetSpec;
    const names = findSpecKeys(dataSpec, 'Boolean');
    return names;
  }

  /** Get the names of all dense features (boolean, categorical, and numeric) */
  @computed
  get denseFeatureNames(): string[] {
    return [...this.categoricalFeatureNames, ...this.numericalFeatureNames,
            ...this.booleanFeatureNames];
  }

  /**
   * Get the names of categorical features, and their possible values.
   */
  @computed
  get categoricalFeatures(): CategoricalFeatures {
    const categoricalFeatures: CategoricalFeatures = {};
    for (const name of this.categoricalFeatureNames) {
      const vocab = this.appState.currentDatasetSpec[name].vocab;
      if (vocab != null) {
        // Use specified vocabulary, if available.
        categoricalFeatures[name] = [...vocab];
      } else {
        // Otherwise, find unique values from the data.
        const uniqueValues = new Set(this.dataService.data.map(
            (d: DataForInput) => d.get(name)!.value));
        categoricalFeatures[name] = [...uniqueValues];
      }
    }
    return categoricalFeatures;
  }

  /**
   * Get the names of numeric features, and their min and max values.
   */
  @computed
  get numericalFeatureRanges(): NumericFeatures {
    const numericFeatures: NumericFeatures = {};
    for (const feat of this.numericalFeatureNames) {
      const values = this.dataService.data.map((d: DataForInput) => {
        return d.get(feat)!.value;
      });
      const min = Math.min(...values);
      const max = Math.max(...values);
      numericFeatures[feat] = [min, max];
    }
    return numericFeatures;
  }

  private numericBinKey(start: number, end: number, isLast: boolean) {
    return `[${start}, ${end}${isLast ? ']' : ')'}`;
  }

  private numericBinRange(start: number, end: number, isLast: boolean) {
    return [start, end + (isLast ? 1e-6 : 0)];
  }

  private freedmanDiaconisBins(feat: string): NumericBins {
    const min = this.numericalFeatureRanges[feat][0];
    const max = this.numericalFeatureRanges[feat][1];
    const values = this.dataService.data.map((d: DataForInput) => {
        return d.get(feat)!.value;
      });
    // The number of bins that the domain is divided into is specified by the
    // FreedmanDiaconis algorithm. The first bin.x0 is always equal to the
    // minimum domain value, and the last bin.x1 is always equal to the
    // maximum domain value. Fall back to a sensible default if the algorithm
    // returns an invalid value.
    let numBins = d3.thresholdFreedmanDiaconis(values, min, max);
    if (numBins === 0 || !isFinite(numBins)) {
      numBins = 10;
    }
    const generator = d3.histogram<number, number>()
        .domain([min, max])
        .thresholds(numBins);
    const generatedBins = generator(values);
    const bins: NumericBins = {};

    for (const bin of generatedBins) {
      const isLastBin = bin.x1 === max;
      const start = roundToDecimalPlaces(bin.x0!, 3);
      const end = roundToDecimalPlaces(bin.x1!, 3);
      // Return if there's an error in the histogram generator and the bin
      // object doesn't contain valid boundary numbers.
      if (start == null || end == null) continue;
      const range = this.numericBinRange(start, end, isLastBin);
      const key = this.numericBinKey(start, end, isLastBin);
      bins[key] = range;
    }

    return bins;
  }

  private discreteBins(feat:string): NumericBins {
    const bins: NumericBins = {};
    const [min, max] = this.numericalFeatureRanges[feat];
    const {step} = this.appState.currentDatasetSpec[feat];

    if (typeof step !== 'number') {
      throw (new Error(
          `Unable to generate discrete bins; '${feat}' step is not defined`));
    }

    for (let lower = min; lower < max; lower += step) {
      const upper = lower + step;
      const binKey = this.numericBinKey(lower, upper, upper === max);
      bins[binKey] = [lower, upper];
    }

    return bins;
  }

  private equalIntervalBins(feat: string, numBins: number): NumericBins {
    const bins: NumericBins = {};
    const min = this.numericalFeatureRanges[feat][0];
    const max = this.numericalFeatureRanges[feat][1];
    const step = (max - min) / numBins;

    for (let i = 0; i < numBins; i++) {
      const start = roundToDecimalPlaces(min + step * i, 3);
      const end = roundToDecimalPlaces(min + step * (i + 1), 3);
      const range = this.numericBinRange(start, end, (i === (numBins - 1)));
      const key = this.numericBinKey(start, end, (i === (numBins - 1)));
      bins[key] = range;
    }

    return bins;
  }

  private quantileBins(feat: string, numBins: number): NumericBins {
    const bins: NumericBins = {};
    const values = this.dataService.data.map(
        (d: DataForInput) => d.get(feat)!.value).sort();
    numBins = Math.min(numBins, values.length);
    const step = values.length / numBins;

    for (let i = 0; i < numBins; i++) {
      const isLast = i === (numBins - 1);
      const start = step * i;
      const end = isLast ? values.length - 1 : step * (i + 1);
      const lower = roundToDecimalPlaces(values[start], 3);
      const upper = roundToDecimalPlaces(values[end], 3);
      const range = this.numericBinRange(lower, upper, isLast);
      const key = this.numericBinKey(lower, upper, isLast);
      bins[key] = range;
    }

    return bins;
  }

  private thresholdBins(feat: string, threshold: number): NumericBins {
    const [min, max] = this.numericalFeatureRanges[feat];
    const [rMin, rMax] = [min, max].map(v => roundToDecimalPlaces(v, 3));
    const rThresh = roundToDecimalPlaces(threshold, 3);
    return {
      [this.numericBinKey(rMin, rThresh, false)]: [min, threshold - 1e-6],
      [this.numericBinKey(rThresh, rMax, true)]: [threshold, max + 1e-6]
    };
  }

  /**
   * Converts a list of FacetingConfigs into a string, ordered by featureName.
   */
  facetConfigsToKey(configs: FacetingConfig[]): string {
    const validConfigs = configs.sort((a, b) => {
      if (a.featureName < b.featureName) { return -1; }
      if (a.featureName > b.featureName) { return 1; }
      return 0;
    });
    return JSON.stringify(validConfigs);
  }

  /**
   * Determines if a faceting configuration is valid. This does not determine if
   * the config is appropriate for the feature type.
   *
   * Rules for different method values:
   *
   * *  DISCRETE and EQUAL_INTERVAL require nothing
   * *  QUANTILE requires numBins is a positive number
   * *  THRESHOLD requries threshold is provided
   */
  validateFacetingConfig(config:FacetingConfig): boolean {
    const {featureName, method, numBins, threshold} = config;
    const isDiscrete = (method == null || method === FacetingMethod.DISCRETE);
    const isEqInt = method === FacetingMethod.EQUAL_INTERVAL;
    const isQuant = method === FacetingMethod.QUANTILE &&
                    typeof numBins === 'number' && numBins > 0;
    const isThreshold = method === FacetingMethod.THRESHOLD &&
                        typeof threshold === 'number';
    const isValid = isDiscrete || isEqInt || isQuant || isThreshold;

    if (!isValid) {
      const confstr = JSON.stringify(config);
      throw new Error(
        `Invalid faceting config for '${featureName}': ${confstr}`);
    }

    return isValid;
  }

  /**
   * Filters a list of faceting configurations to only those that are valid.
   */
  validateFacetingConfigs(configs:FacetingConfig[]): FacetingConfig[] {
    return configs.filter(this.validateFacetingConfig);
  }

  /**
   * Get the names of numeric features, and their bins. This relies on
   * numericalFeatureRanges.
   */
  numericalFeatureBins(configs: FacetingConfig[]): NumericFeatureBins {
    const featureBins: NumericFeatureBins = {};
    const numericConfigs = configs
        .filter(c => this.numericalFeatureNames.includes(c.featureName));

    for (const config of numericConfigs) {
      const {featureName, method, numBins, threshold} = config;
      if (method === FacetingMethod.EQUAL_INTERVAL) {
        featureBins[featureName] = (numBins == null || numBins < 1) ?
            this.freedmanDiaconisBins(featureName) :
            this.equalIntervalBins(featureName, numBins);
      } else if (method === FacetingMethod.QUANTILE) {
        featureBins[featureName] = this.quantileBins(featureName, numBins || 4);
      } else if (method === FacetingMethod.THRESHOLD && threshold != null) {
        featureBins[featureName] = this.thresholdBins(featureName, threshold);
      } else {
        featureBins[featureName] = this.discreteBins(featureName);
      }
    }

    return featureBins;
  }

  /**
   * Find the correct feature bin for this input. Returns the bin values, or
   * null if the datapoint should not be in any of the bins.
   */
  getNumericalBinForExample(
    bins: NumericFeatureBins, input: IndexedInput, feature: string): number[] | null {
    const value = input.data[feature];
    for (const bin of Object.values(bins[feature])) {
      const [start, end] = bin;
      if (start <= value && value < end) { return bin; }
    }
    return null;
  }

  /**
   * Create a label to tell the user how many intersectional groups there are
   * between all possible values of the given features (e.g., "6 x 2 = 12") if
   * there are two features, with 6 and 2 values respectively.
   */
  numIntersectionsLabel(bins: NumericFeatureBins, features: string[]): string {
    const numLabels = features.map(feature => {
      if (this.categoricalFeatureNames.includes(feature)) {
        return this.categoricalFeatures[feature].length;
      }
      if (this.numericalFeatureNames.includes(feature)) {
        return Object.keys(bins[feature]).length;
      }
      if (this.booleanFeatureNames.includes(feature)) {
        return 2;
      }
      return 0;
    });
    const total = numLabels.reduce((a, b) => a * b);
    return numLabels.length > 1 ? `${numLabels.join('x')} = ${total}` :
                                  numLabels[0].toString();
  }

  /**
   * Given a set of IndexedInputs and a set of features (each of which has a set
   * of possible values or bins), organize the datapoints by all intersectional
   * combinations of the these, and return a dict of keys to groups.
   *
   * By default, assumes that the features are IndexedInput data features.
   * However, an optional getFeatValForInput function can be provided to use
   * this with arbitrary features (e.g., predicted values as in the
   * matrix_module.)
   */
  groupExamplesByFeatures(
      bins: NumericFeatureBins, data: IndexedInput[], features: string[],
      getFeatValForInput: GetFeatureFunc = (b, d, i, f) =>
          this.getFeatureValForInput(b, d, f)): GroupedExamples {
    const facetedData: GroupedExamples = {};

    // Loop over each datapoint to see what bin it belongs into.
    for (let i = 0; i < data.length; i++) {
      // Filter the data features for those that we are faceting by.
      const datum = data[i];
      const dFilters: FacetMap = {};

      for (const feature of features) {
        const val = getFeatValForInput(bins, datum, i, feature);
        if (val == null) { continue; }
        if (Array.isArray(val)) {   // Numeric features are number[], need key
          const binIdx = Object.values(bins[feature]).indexOf(val);
          const displayVal = Object.keys(bins[feature])[binIdx];
          dFilters[feature] = {val, displayVal};
        } else {                    // Otherwise, val is the displayVal
          dFilters[feature] = {val, displayVal: val.toString()};
        }
      }

      // Make a dictionary key from this set of features.
      const comboKey = facetMapToDictKey(dFilters);

      // If there haven't been any other datapoints with this combination of
      // filters, start a new facet.
      if (!facetedData.hasOwnProperty(comboKey)) {
        facetedData[comboKey] = {
          displayName: comboKey,
          data: [],
          facets: dFilters
        };
      }
      facetedData[comboKey].data.push(datum);
    }

    return facetedData;
  }

  /**
   * Get the feature value for a datapoint. Will return the binned value,
   * if the datapoint is numerical. Will return null if the datapoint does
   * not have the feature.
   */
  getFeatureValForInput(
    bins: NumericFeatureBins, d: IndexedInput, feature: string): string | null {
    if (feature in d.data) {
      const isNumerical = this.numericalFeatureNames.includes(feature);
      return isNumerical ? this.getNumericalBinForExample(bins, d, feature) :
                           d.data[feature];
    }
    return null;
  }
}
