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
 * Tests for frontend caching helpers.
 */

import 'jasmine';

import {BatchRequestCache} from './caching';
import {cleanState} from './testing_utils';
import {range} from './utils';

class DummyModel {
  shouldFail: boolean = false;

  predict(input: number): string {
    return `f(${input})`;
  }

  async predictBatch(inputs: number[]): Promise<string[]|null> {
    if (this.shouldFail) {
      return null;
    }
    return inputs.map((i) => this.predict(i));
  }
}

describe('BatchRequestCache test', () => {
  const state = cleanState(() => {
    const model = new DummyModel();
    spyOn(model, 'predict').and.callThrough();
    spyOn(model, 'predictBatch').and.callThrough();
    const cache = new BatchRequestCache<string, number, string>(
        async (inputs: number[]) => model.predictBatch(inputs),
        (input: number) => `key(${input})`);
    return {model, cache};
  });

  it('is behaving correctly via spyOn', async () => {
    // Verify that spyOn doesn't change model output to 'undefined',
    // because this could cause incorrect behavior in other tests.
    const inputs = range(10);
    const modelOutputs = await state.model.predictBatch(inputs);
    expect(modelOutputs).toEqual(inputs.map((input: number) => `f(${input})`));
  });

  it('matches on first request', async () => {
    // Call on initial batch
    const firstInputs = range(10);
    const firstCallOutputs = await state.cache.call(firstInputs);

    // Check call counts.
    expect(state.model.predict).toHaveBeenCalledTimes(firstInputs.length);
    expect(state.model.predictBatch).toHaveBeenCalledTimes(1);

    // Verify cached call matches a direct call.
    expect(firstCallOutputs)
        .toEqual(await state.model.predictBatch(firstInputs));
  });

  it('matches disjoint second request', async () => {
    // Call on initial batch
    const firstInputs = range(10);
    await state.cache.call(firstInputs);

    // Call on second, disjoint batch
    const secondInputs = [51, 52, 53];
    const secondCallOutputs = await state.cache.call(secondInputs);

    // Check call counts.
    expect(state.model.predict)
        .toHaveBeenCalledTimes(firstInputs.length + secondInputs.length);
    expect(state.model.predictBatch).toHaveBeenCalledTimes(2);

    // Verify cached call matches a direct call.
    expect(secondCallOutputs)
        .toEqual(await state.model.predictBatch(secondInputs));
  });

  it('matches intersecting second request', async () => {
    // Call on initial batch
    const firstInputs = range(10);
    await state.cache.call(firstInputs);

    // Call on second, disjoint batch
    const secondInputs = [4, 6, 8, 65, 66, 67];
    const secondCallOutputs = await state.cache.call(secondInputs);

    // Check call counts. Should only call predict() for new inputs.
    expect(state.model.predict).toHaveBeenCalledTimes(firstInputs.length + 3);
    expect(state.model.predictBatch).toHaveBeenCalledTimes(2);

    // Verify cached call matches a direct call.
    expect(secondCallOutputs)
        .toEqual(await state.model.predictBatch(secondInputs));
  });

  it('matches contained second request', async () => {
    // Call on initial batch
    const firstInputs = range(10);
    await state.cache.call(firstInputs);

    // Call on second, disjoint batch
    const secondInputs = [4, 6, 8];
    const secondCallOutputs = await state.cache.call(secondInputs);

    // Check call counts. Should only call predict() for new inputs.
    expect(state.model.predict).toHaveBeenCalledTimes(firstInputs.length);
    // Cache should never call model if everything hits, to avoid
    // unnecessary round-trip latency.
    expect(state.model.predictBatch).toHaveBeenCalledTimes(1);

    // Verify cached call matches a direct call.
    expect(secondCallOutputs)
        .toEqual(await state.model.predictBatch(secondInputs));
  });

  it('handles failed request', async () => {
    // Call on initial batch
    const firstInputs = range(10);
    await state.cache.call(firstInputs);

    state.model.shouldFail = true;

    // Call on second batch. At least one miss, so backend will fail.
    const secondInputs = [8, 9, 10];
    const secondCallOutputs = await state.cache.call(secondInputs);
    expect(secondCallOutputs).toEqual(null);

    // Call on third batch. Cache hits, so backend never fails.
    const thirdInputs = [4, 6, 8];
    const thirdCallOutputs = await state.cache.call(thirdInputs);
    state.model.shouldFail = false;  // disable now to get real output
    expect(thirdCallOutputs)
        .toEqual(await state.model.predictBatch(thirdInputs));
  });
});
