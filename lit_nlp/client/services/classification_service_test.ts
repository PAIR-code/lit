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

import 'jasmine';

import {IndexedInput, Spec} from '../lib/types';

import {getPredictionClass, MarginsPerField} from './classification_service';


describe('getPredictionClass test', () => {
  const predKey = 'key';
  const outputSpec: Spec = {};
  outputSpec[predKey] = {
    __class__: 'LitType',
    __name__: 'TextSegment',
    __mro__: ['TextSegment', 'LitType', 'object'],
    null_idx: 0
  };
  const mockInput: IndexedInput = {
      id: 'xxxxxxx',
      data: {'testFeat0': 1, 'testNumFeat0': 0},
      meta: {}
    };

  it('gets prediction class index given 2 classes and 0 margin', () => {
    const margins: MarginsPerField = {};
    margins[predKey] = {"": {margin: 0}};

    let scores = [1, 0];
    let predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [0, 1];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(1);

    // Tests near decision boundary.
    scores = [.5, .5];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.5 + 1e-4, .5 - 1e-4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.5 - 1e-4, .5 + 1e-4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(1);
  });

  it('gets prediction class index given 2 classes with non-zero margin', () => {
    const margins: MarginsPerField = {};

    // The margin can be calculated from the binary threshold using:
    // -ln(1/threshold - 1).
    const marginFromThreshold = (t: number) => -1 * Math.log(1 / t - 1);
    margins[predKey] = {"": {margin: marginFromThreshold(.6)}};

    // Tests near decision boundary.
    let scores = [.4, .6];
    let predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.4 + 1e-4, .6 - 1e-4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.4 - 1e-1, .6 + 1e-1];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(1);
  });

  it('gets prediction class index given 3 classes and 0 margin', () => {
    const margins: MarginsPerField = {};
    margins[predKey] = {"": {margin: 0}};

    let scores = [.1, .3, .6];
    let predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(2);

    scores = [.2, .4, .4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(1);

    scores = [.3333, .3333, .3333];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);
  });

  it('gets prediction class index given 3 classes and non-zero margin', () => {
    const margins: MarginsPerField = {};

    // Margin at which null class score is equal to the max class score:
    // ln(max class score) - ln(null class score)
    const maxScoreMargin = (scores: number[]) => {
      return Math.log(Math.max(...scores)) - Math.log(scores[0]);
    };

    // Tests near decision boundary.
    let scores = [.1, .3, .6];
    margins[predKey] = {"": {margin: maxScoreMargin(scores)}};

    let predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.1 + 1e-4, .3, .6 - 1e-4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.1 - 1e-4, .3, .6 + 1e-4];
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(2);


    // Testing extreme margin values.
    scores = [.01, .98, .01];
    margins[predKey] = {"": {margin: 5}};
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(0);

    scores = [.98, .005, .015];
    margins[predKey] = {"": {margin: -5}};
    predictionClass = getPredictionClass(
        scores, predKey, outputSpec, mockInput, undefined, margins);
    expect(predictionClass).toBe(2);
  });
});
