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
import {findSpecKeys, objToDictKey, roundToDecimalPlaces} from '../lib/utils';

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

/**
 * A map of numeric features to their categorical bins.
 */
export interface NumericFeatureBins {
  [feature: string]: {[key: string]: number[]};
}

/**
 * Function type for getting a (possibly binned) feature value from an
 * IndexedInput. Inputs are the datapoint, its index in the current data,
 * and the feature key to get the value for.
 */
export type GetFeatureFunc = (d: IndexedInput, i: number, key: string) =>
    number|string|null;

/**
 * A singleton class that handles grouping.
 */
export class GroupService extends LitService {
  constructor(private readonly appState: AppState) {
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
    const dataSpec = this.appState.currentDatasetSpec;
    const names = findSpecKeys(dataSpec, 'Scalar');
    return names;
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
        const uniqueValues = new Set(this.appState.currentInputData.map(
            (d: IndexedInput) => d.data[name]));
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
    this.numericalFeatureNames.forEach(feat => {
      const values = this.appState.currentInputData.map((d: IndexedInput) => {
        return d.data[feat];
      });
      const min = Math.min(...values);
      const max = Math.max(...values);
      numericFeatures[feat] = [min, max];
    });
    return numericFeatures;
  }

  /**
   * Get the names of numeric features, and their bins. This relies on
   * numericalFeatureRanges.
   */
  @computed
  get numericalFeatureBins() {
    const ranges: NumericFeatureBins = {};
    this.numericalFeatureNames.forEach(feat => {
      const values = this.appState.currentInputData.map((d: IndexedInput) => {
        return d.data[feat];
      });
      const min = this.numericalFeatureRanges[feat][0];
      const max = this.numericalFeatureRanges[feat][1];

      // The number of bins that the domain is divided into is specified by the
      // FreedmanDiaconis algorithm. The first bin.x0 is always equal to the
      // minimum domain value, and the last bin.x1 is always equal to the
      // maximum domain value. Fall back to a sensible default if the algorithm
      // returns an invalid valid.
      let numBins = d3.thresholdFreedmanDiaconis(values, min, max);
      if (numBins === 0 || !isFinite(numBins)) {
        numBins = 10;
      }
      const generator: d3.HistogramGeneratorNumber<number, number> =
          d3.histogram<number, number>()
              .domain([min, max])
              .thresholds(numBins);
      const generatedBins = generator(values);
      const keyToRanges: {[name: string]: number[]} = {};

      generatedBins.forEach((bin, i) => {
        const start = roundToDecimalPlaces(bin.x0!, 3);
        const end = roundToDecimalPlaces(bin.x1!, 3);
        // Return if there's an error in the histogram generator and the bin
        // object doesn't contain valid boundary numbers.
        if (start == null || end == null) return;

        let range = [start, end];
        let key = `[${start}, ${end})`;

        // TODO(ellenj): Change this not to depend on the generated bins being
        // in a certain order where the largest bin is always the last bin. If
        // this is the last bin in generatedBins, include the upper bound.
        if (i === (generatedBins.length - 1)) {
          range = [start, end + 1e-6];
          key = `[${start}, ${end}]`;
        }
        keyToRanges[key] = range;
      });
      ranges[feat] = keyToRanges;
    });
    return ranges;
  }

  /**
   * Find the correct feature bin for this input. Returns the dict key, or null
   * if the datapoint should not be in any of the bins.
   */
  private getNumericalBinKeyForExample(input: IndexedInput, feat: string):
      string|null {
    const range = this.numericalFeatureBins[feat];
    for (const key of Object.keys(range)) {
      const start = range[key][0];
      const end = range[key][1];
      const featureValue = input.data[feat];
      if (featureValue >= start && featureValue < end) {
        return key;
      }
    }
    return null;
  }

  /**
   * Create a label to tell the user how many intersectional groups there are
   * between all possible values of the given features (e.g., "6 x 2 = 12") if
   * there are two features, with 6 and 2 values respectively.
   */
  numIntersectionsLabel(features: string[]): string {
    const numLabels = features.map(v => {
      if (this.categoricalFeatureNames.includes(v)) {
        return this.categoricalFeatures[v].length;
      }
      if (this.numericalFeatureNames.includes(v)) {
        return Object.keys(this.numericalFeatureBins[v]).length;
      }
      if (this.booleanFeatureNames.includes(v)) {
        return 2;
      }
      return 0;
    });
    const total = numLabels.reduce((a, b) => a * b);
    return numLabels.length > 1 ? `${numLabels.join('x')} = ${total}` :
                                  numLabels[0].toString();
  }

  /**
   * Given a set of IndexedInputs and a set of features (each of which has a
   * set of possible values or bins), organize the datapoints by all intersectional
   * combinations of the these, and return a dict of keys to groups.
   *
   * By default, assumes that the features are IndexedInput data features.
   * However. an optional getFeatValForInput function can be provided to use
   * this with arbitrary features (e.g., predicted values as in the
   * matrix_module.)
   */
  groupExamplesByFeatures(
      data: IndexedInput[], features: string[],
      getFeatValForInput: GetFeatureFunc = (d, i, key) =>
          this.getFeatureValForInput(d, i, key)): GroupedExamples {
    const facetedData: GroupedExamples = {};

    // Loop over each datapoint to see what bin it belongs into.
    data.forEach((d, i) => {
      // Filter the data features for those that we are faceting by.
      const dFilters: FacetMap = {};
      features.forEach(key => {
        const featValForData = getFeatValForInput(d, i, key);
        if (featValForData != null) {
          dFilters[key] = featValForData;
        }
      });

      // Make a dictionary key from this set of features.
      const comboKey = objToDictKey(dFilters);

      // If there haven't been any other datapoints with this combination of
      // filters, start a new facet.
      if (!facetedData.hasOwnProperty(comboKey)) {
        facetedData[comboKey] = {
          displayName: comboKey,
          data: [],
          facets: dFilters
        };
      }
      facetedData[comboKey].data.push(d);
    });
    return facetedData;
  }

  /**
   * Get the feature value for a datapoint. Will return the binned value,
   * if the datapoint is numerical. Will return null if the datapoint does
   * not have the feature.
   */
  getFeatureValForInput(d: IndexedInput, i: number, key: string): string|null {
    if (key in d.data) {
      const isNumerical = this.numericalFeatureNames.includes(key);
      return isNumerical ? this.getNumericalBinKeyForExample(d, key) : d.data[key];
    }
    return null;
  }
}
