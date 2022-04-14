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
import {action, computed, observable, reaction} from 'mobx';

import {ColorOption, D3Scale, FacetedData, GroupedExamples, IndexedInput, Preds, Spec} from '../lib/types';
import {findSpecKeys} from '../lib/utils';
import {BINARY_POS_NEG, CATEGORICAL_NORMAL} from '../lib/colors';

import {LitService} from './lit_service';
import {ApiService, AppState, GroupService} from './services';
import {FacetingConfig, FacetingMethod} from './group_service';

interface AllClassificationInfo {
  [id: string]: PerExampleClassificationInfo;
}

interface PerExampleClassificationInfo {
  [model: string]: PerExamplePerModelClassificationInfo;
}

interface PerExamplePerModelClassificationInfo {
  [predKey: string]: ClassificationInfo;
}

/**
 * Info about individual classifications including computed properties.
 * These interface field names should be in sync with the
 * fieldsToDisplayNames Map.
 */
export interface ClassificationInfo {
  predictions: number[];
  predictedClassIdx: number;
  predictionCorrect?: boolean;
}

/**
 * A map from classification info field names to their display names.
 */
const classificationDisplayNames = new Map([
  ['predictions', 'predictions'],
  ['predictedClassIdx', 'predicted class index'],
  ['predictionCorrect', 'prediction correct'],
]);

/**
 * A margin setting is the margin value and the facet information for which
 * datapoints from a dataset that margin value applies to.
 */
interface MarginSetting {
  facetData?: FacetedData;
  margin: number;
}

/**
 * Any facet of a dataset can have its own margin value. Key string represents
 * the facet key/value pairs.
 */
interface MarginsPerFacet {
  [facetString: string]: MarginSetting;
}

/** Each output field has its own margin settings. */
export interface MarginsPerField {
  [fieldName: string]: MarginsPerFacet;
}

/**
 * Classification margin settings across all models and prediction heads.
 *
 * Margins are a generalized way to define classification thresholds beyond
 * binary classification score threshoods.
 */
export interface MarginSettings {
  [model: string]: MarginsPerField;
}

/**
 * Given an example and the margins for a field, return the appropriate margin
 * to use for the example.
 */
function getMarginSettingForExample(
    input: IndexedInput, marginsPerFacet: MarginsPerFacet,
    groupService?: GroupService) {
  // If there is an empty string entry, this represents the margin for the
  // entire dataset.
  if ("" in marginsPerFacet) {
    return marginsPerFacet[""];
  }
  // Find the facet that matches the example provided.
  for (const group of Object.values(marginsPerFacet)) {
    let matches = true;
    for (const field of Object.keys(group.facetData!.facets!)) {
      if (groupService != null &&
          groupService.numericalFeatureNames.includes(field)) {
        const facetConfig: FacetingConfig = {
          featureName: field,
          method: FacetingMethod.EQUAL_INTERVAL,
          numBins: 0
        };
        const bins = groupService.numericalFeatureBins([facetConfig]);
        const bin = groupService.getNumericalBinForExample(bins, input, field)!;
        const groupRange = group.facetData!.facets![field].val as number[];
        if (bin[0] !== groupRange[0] || bin[1] !== groupRange[1]) {
          matches = false;
          break;
        }
      } else if (input.data[field] !== group.facetData!.facets![field].val) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return group;
    }
  }
  return null;
}

/**
 * Return the predicted class index given prediction scores and settings.
 */
export function getPredictionClass(
    scores: number[], predKey: string, outputSpec: Spec, input: IndexedInput,
    groupService?: GroupService, margins?: MarginsPerField) {
  let margin = 0;
  if (margins?.[predKey] != null) {
    const group = getMarginSettingForExample(
        input, margins[predKey], groupService);
    if (group != null) {
      margin = margins[predKey][group.facetData?.displayName || ""].margin;
    }
  }
  const nullIdx = outputSpec[predKey].null_idx;
  let maxScore = -Infinity;
  let maxIndex = 0;
  // Find max of the log prediction scores, adding any provided margin
  // to the null class, if there is one set.
  for (let i = 0; i < scores.length; i++) {
    let score = Math.log(scores[i]);
    if (nullIdx === i) {
      score += margin;
    }
    if (maxScore < score) {
      maxScore = score;
      maxIndex = i;
    }
  }
  return maxIndex;
}

/**
 * A singleton class that handles calculating and storing per-input
 * classification response information.
 */
export class ClassificationService extends LitService {
  @observable classificationInfo: AllClassificationInfo = {};
  @observable marginSettings: MarginSettings = {};

  // Stores list of possible labels for a given model/prediction-key
  // combination.
  @observable
  private readonly labelNames: {[modelAndKey: string]: string[]} = {};

  constructor(
      private readonly apiService: ApiService,
      private readonly appState: AppState,
      private readonly groupService: GroupService) {
    super();
    reaction(() => this.allMarginSettings, margins => {
      this.updateClassifications();
    });
    reaction(() => appState.currentModels, currentModels => {
      this.reset();
    });
  }

  // Returns all margin settings for use as a reaction input function when
  // setting up observers.
  // TODO(lit-team): Remove need for this intermediate object (b/156100081)
  @computed
  get allMarginSettings(): number[] {
    const res: number[] = [];
    for (const settingsPerModel of Object.values(this.marginSettings)) {
      for (const settingsPerPredKey of Object.values(settingsPerModel)) {
         for (const settingsPerFacet of Object.values(settingsPerPredKey)) {
           res.push(settingsPerFacet.margin);
         }
      }
    }
    return res;
  }

  /**
   * Reset the facet groups that store margins for a field based on the
   * facets from the groupedExamples.
   */
  @action
  setMarginGroups(model: string, fieldName: string,
                  groupedExamples: GroupedExamples) {
    if (this.marginSettings[model] == null) {
      this.marginSettings[model] = {};
    }
    this.marginSettings[model][fieldName] = {};
    for (const group of Object.values(groupedExamples)) {
      this.marginSettings[model][fieldName][group.displayName!] =
          {facetData: group, margin: 0};
    }
  }

  @action
  setMargin(model: string, fieldName: string, value: number,
            facet?: FacetedData) {
    if (this.marginSettings[model] == null) {
      this.marginSettings[model] = {};
    }
    if (this.marginSettings[model][fieldName] == null) {
      this.marginSettings[model][fieldName] = {};
    }
    if (facet == null) {
      // If no facet provided, then update the facet for the entire dataset
      // if one exists, otherwise update all facets with the provided margin.
      if ("" in this.marginSettings[model][fieldName]) {
        this.marginSettings[model][fieldName][""] =
            {facetData: facet, margin: value};
      } else {
        for (const key of Object.keys(this.marginSettings[model][fieldName])) {
          this.marginSettings[model][fieldName][key].margin = value;
        }
      }
    } else {
      this.marginSettings[model][fieldName][facet.displayName!] =
          {facetData: facet, margin: value};
    }
  }

  getMargin(model: string, fieldName: string, facet?: FacetedData) {
    if (this.marginSettings[model] == null ||
        this.marginSettings[model][fieldName] == null) {
      return 0;
    }
    if (facet == null) {
      if (this.marginSettings[model][fieldName][""] == null) {
        return 0;
      }
      return this.marginSettings[model][fieldName][""].margin;
    } else {
      if (this.marginSettings[model][fieldName][facet.displayName!] == null) {
        return 0;
      }
      return this.marginSettings[model][fieldName][facet.displayName!].margin;
    }
  }

  /**
   * Calls the server to get multiclass predictions and calculate related info.
   * @param inputs inputs to run model on
   * @param model model to query
   * @param datasetName current dataset (for caching)
   */
  async getClassificationPreds(
      inputs: IndexedInput[], model: string,
      datasetName: string): Promise<Preds[]> {
    // TODO(lit-team): Use client-side cache when available.
    const result = this.apiService.getPreds(
        inputs, model, datasetName, ['MulticlassPreds']);
    const preds = await result;
    if (preds != null && preds.length > 0) {
      this.processNewPreds(inputs, model, preds);
    }
    return result;
  }

  /**
   * Get labels list for a given prediction task.
   */
  getLabelNames(model: string, predKey: string) {
    return this.labelNames[`${model}:${predKey}`];
  }

  getInfoFields() {
    return Array.from(classificationDisplayNames.keys());
  }

  getDisplayNames() {
    return Array.from(classificationDisplayNames.values());
  }

  private updateClassifications() {
    for (const input of this.appState.currentInputData) {
      const info = this.classificationInfo[input.id];
      if (info == null) {
        continue;
      }
      const models = Object.keys(info);
      for (const model of models) {
        const predKeys = Object.keys(info[model]);
        for (const predKey of predKeys) {
          const fields = info[model][predKey];
          this.updateClassification(fields, input, predKey, model);
        }
      }
    }
  }

  private updateClassification(
      fields: ClassificationInfo, input: IndexedInput, predKey: string,
      model: string) {
    const outputSpec = this.appState.currentModelSpecs[model].spec.output;
    fields.predictedClassIdx = getPredictionClass(
        fields.predictions, predKey, outputSpec, input, this.groupService,
        this.marginSettings[model]);
    // If there are labels, use those. Otherwise just use prediction
    // array indices.
    const labelKey = `${model}:${predKey}`;
    if (this.labelNames[labelKey] == null) {
      this.labelNames[labelKey] = outputSpec[predKey].vocab ||
          Array.from({length: fields.predictions.length}, (v, k) => `${k}`);
    }
    const labelField = outputSpec[predKey].parent;
    if (labelField != null) {
      fields.predictionCorrect = input.data[labelField] ===
          this.labelNames[labelKey][fields.predictedClassIdx];
    }
  }

  /**
   * Reset stored info. Used when active models change.
   */
  reset() {
    this.classificationInfo = {};
  }

  /**
   * Gets stored results for the given datapoints and predictions.
   */
  async getResults(ids: string[], model: string, predKey: string):
      Promise<ClassificationInfo[]> {
    // TODO(lit-dev): rate-limit this and batch requests, otherwise if this is
    // called on individual datapoints it will make a massive number of backend
    // requests.

    const unstoredIds: string[] = [];

    ids.forEach((id) => {
      if (this.classificationInfo[ids[0]]?.[model]?.[predKey] == null) {
        unstoredIds.push(id);
      }
    });

    // If any results aren't yet stored in the front-end, then get them.
    if (unstoredIds.length > 0) {
      await this.getClassificationPreds(
          this.appState.getExamplesById(unstoredIds), model,
          this.appState.currentDataset);
    }

    const results: ClassificationInfo[] = [];
    ids.forEach((id) => {
      const classificationInfo =
          this.classificationInfo[id]?.[model]?.[predKey];
      if (classificationInfo != null) {
        results.push(classificationInfo);
      }
    });
    return results;
  }

  private processNewPreds(
      inputs: IndexedInput[], model: string, preds: Preds[]) {
    const outputSpec = this.appState.currentModelSpecs[model].spec.output;
    const multiclassKeys = findSpecKeys(outputSpec, 'MulticlassPreds');
    const predictedKeys = Object.keys(preds[0]);

    for (let i = 0; i < preds.length; i++) {
      const input = inputs[i];
      const pred = preds[i];
      if (this.classificationInfo[input.id] == null) {
        this.classificationInfo[input.id] = {} as PerExampleClassificationInfo;
      }
      if (this.classificationInfo[input.id][model] == null) {
        this.classificationInfo[input.id][model] = {} as
            PerExamplePerModelClassificationInfo;
      }
      for (let predIndex = 0; predIndex < predictedKeys.length; predIndex++) {
        const predKey = predictedKeys[predIndex];
        if (!multiclassKeys.includes(predKey)) {
          continue;
        }
        const fields = {} as ClassificationInfo;
        fields.predictions = pred[predKey] as number[];
        this.updateClassification(fields, input, predKey, model);
        if (this.classificationInfo[input.id][model][predKey] == null) {
          this.classificationInfo[input.id][model][predKey] = fields;
        }
      }
    }
  }

  @computed
  get colorOptions(): ColorOption[] {
    const ids = Object.keys(this.classificationInfo);
    if (ids.length === 0) {
      return [];
    }
    const options: ColorOption[] = [];
    const info = this.classificationInfo[ids[0]];
    const models = Object.keys(info);
    for (const model of models) {
      const predKeys = Object.keys(info[model]);
      for (const predKey of predKeys) {
        options.push({
          name: `${model}:${predKey} class`,
          getValue: (input: IndexedInput) =>
              this.labelNames[`${model}:${predKey}`]
                             [this.classificationInfo[input.id][model][predKey]
                                  .predictedClassIdx],
          scale: d3.scaleOrdinal(CATEGORICAL_NORMAL)
                   .domain(this.labelNames[`${model}:${predKey}`]) as D3Scale
        });
        if (info[model][predKey].predictionCorrect != null) {
          options.push({
            name: `${model}:${predKey} correct`,
            getValue: (input: IndexedInput) =>
                this.classificationInfo[input.id][model][predKey]
                    .predictionCorrect ? 'correct' : 'incorrect',
            scale: d3.scaleOrdinal(BINARY_POS_NEG)
                     .domain(['correct', 'incorrect']) as D3Scale
          });
        }
      }
    }
    return options;
  }
}
