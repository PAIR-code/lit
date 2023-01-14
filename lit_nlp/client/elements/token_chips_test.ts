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

import 'jasmine';

import {LitElement} from 'lit';

import {TokenChips} from './token_chips';

describe('token chips test', () => {
  [{
    tokensWithWeights: [{token: 'hello', weight: 1.0}],
  },
   {
     tokensWithWeights:
         [{token: 'hello', weight: 0.7}, {token: 'world', weight: 0.3}],
   }].forEach(({tokensWithWeights}) => {
    let tokenChips: TokenChips;

    beforeEach(async () => {
      tokenChips = new TokenChips();
      tokenChips.tokensWithWeights = tokensWithWeights;
      document.body.appendChild(tokenChips);
      await tokenChips.updateComplete;
    });

    it('should instantiate correctly', () => {
      expect(tokenChips).toBeDefined();
      expect(tokenChips instanceof HTMLElement).toBeTrue();
      expect(tokenChips instanceof LitElement).toBeTrue();
    });

    it('should render a set of token elements', async () => {
      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      expect(tokenElements[0].innerText).toEqual(tokensWithWeights[0].token);
    });
  });
});
