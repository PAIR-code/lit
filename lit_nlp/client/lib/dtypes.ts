/**
 * @license
 * Copyright 2022 Google LLC
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
 * Datatypes for representing structured output used in the front end.
 * Corresponds to ../api/dtypes.py.
 */

// Some class properties are in snake_case to match their Python counterparts.
// tslint:disable:enforce-name-casing

interface DataTuple {}

/** Dataclass for individual span label preds. Can use this in model preds. */
export interface SpanLabel extends DataTuple {
  start: number;  // Inclusive.
  end: number;    // Exclusive.

  label?: string;
  align?: string;  // Name of field (segment) this aligns to.
}

/** Dataclass for individual edge label preds. Can use this in model preds. */
export interface EdgeLabel extends DataTuple {
  span1: [number, number];  // Inclusive, exclusive.
  span2?: [number, number];  // Inclusive, exclusive.
  label: string|number;
}

/** Dataclass for annotation clusters, which may span multiple segments. */
export interface AnnotationCluster extends DataTuple {
  label: string;
  spans: SpanLabel[];
  score?: number;
}

// TODO(b/196886684): document API for salience interpreters.
/** Dataclass for a salience map over tokens. */
export interface TokenSalience extends DataTuple {
  tokens: string[];
  salience: number[];  // Parallel to tokens.
}

/** Dataclass for a salience map over categorical and/or scalar features. */
export interface FeatureSalience extends DataTuple {
  salience: {[key: string]: number};
}

// TODO(b/196886684): document API for salience interpreters.
/** Dataclass for a salience map over a target sequence. */
export interface SequenceSalienceMap extends DataTuple {
  tokens_in: string[];
  tokens_out: string[];
  salience: number[][];
}

/** A tuple of text and its score. */
export type ScoredTextCandidate = [text: string, score: number|null];

/** A list of (text, score) tuples. */
export type ScoredTextCandidates = ScoredTextCandidate[];
