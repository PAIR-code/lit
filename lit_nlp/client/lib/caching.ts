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
 * Caching helpers and classes.
 */

/**
 * Implements a cache for batched requests, i.e. any function that operates
 * independently on each input. The cache stores results for previous requests
 * based on keyFn on each row, and only computes new results for the subset of
 * inputs not found in the cache.
 * Functionally, this is similar to the server-side CachingModelWrapper.
 *
 * TODO(lit-dev): implement a locking mechanism so we don't dispatch multiple
 * requests for the same data. Should block for the entirety of call(), in case
 * an in-flight request can provide the requested values.
 */
export class BatchRequestCache<IdType, InputType, OutputType> {
  private readonly resultCache = new Map<IdType, OutputType>();

  constructor(
      private readonly requestFn:
          (inputs: InputType[]) => Promise<OutputType[]|null>,
      private readonly keyFn: (input: InputType) => IdType) {}

  async call(inputs: InputType[]): Promise<OutputType[]|null> {
    // Find cached values
    const results =
        inputs.map(input => this.resultCache.get(this.keyFn(input)));
    // Indices of any cache misses.
    const missIdxs: number[] = [];
    for (let i = 0; i < results.length; ++i) {
      if (results[i] === undefined) missIdxs.push(i);
    }
    // Short-circuit to avoid calling requestFn.
    if (missIdxs.length === 0) {
      return results as OutputType[];  // should no longer contain 'undefined'
    }

    // Prepare inputs and make a request.
    const requestInputs = missIdxs.map(i => inputs[i]);
    const requestResults = await this.requestFn(requestInputs);
    if (requestResults === null) {
      return null;
    }

    // Merge new results back in, and populate the cache.
    missIdxs.forEach((origIdx, i) => {
      const key = this.keyFn(inputs[origIdx]);
      const value = requestResults[i];
      this.resultCache.set(key, value);
      results[origIdx] = value;
    });
    return results as OutputType[];  // should no longer contain 'undefined'
  }
}
