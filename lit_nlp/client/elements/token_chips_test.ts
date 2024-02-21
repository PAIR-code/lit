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

import {TokenChips, TokenWithWeight} from './token_chips';
import {LitTooltip} from './tooltip';

const TESTDATA: Array<{tokensWithWeights: TokenWithWeight[]}> = [
  {
    tokensWithWeights: [{token: 'hello', weight: 1.0, forceShowTooltip: true}],
  },
  {
    tokensWithWeights: [
      {token: 'hello', weight: 0.7, selected: true, pinned: true},
      {token: 'world', weight: 0.3}
    ],
  },
  {
    // for testing preSpace mode
    tokensWithWeights: [
      {token: 'foo', weight: 0.7, selected: true, pinned: true},
      {token: ' bar', weight: 0.3}, {token: 'baz', weight: 0.5}
    ],
  },
  {
    // for testing breakNewlines mode
    tokensWithWeights: [
      {token: 'foo', weight: 0.7}, {token: '\nbar', weight: 0.3},
      {token: '\n\n', weight: 0.1}, {token: 'baz\n', weight: 0.5}
    ],
  },
];

describe('token chips test', () => {
  TESTDATA.forEach(({tokensWithWeights}) => {
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
      expect(tokenElements[0].children[0]).toBeInstanceOf(LitTooltip);
    });

    it('should break spaces in preSpace mode', async () => {
      tokenChips.preSpace = true;
      await tokenChips.updateComplete;

      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      for (let i = 0; i < tokenElements.length; i++) {
        const elem = tokenElements[i];
        const expectedToken = tokensWithWeights[i].token;
        if (expectedToken.startsWith(' ')) {
          // Space moved to a word spacer.
          expect(elem.innerText).toEqual(expectedToken.slice(1));
          expect(elem.previousElementSibling?.classList ?? [])
              .toContain('word-spacer');
        } else {
          // Space intact, no word spacer.
          expect(elem.innerText).toEqual(expectedToken);
          if (i > 0) {
            expect(elem.previousElementSibling?.classList ?? [])
                .toContain('salient-token');
          }
        }
      }
    });

    it('should break newlines in breakNewlines mode', async () => {
      tokenChips.breakNewlines = true;
      await tokenChips.updateComplete;

      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      for (let i = 0; i < tokenElements.length; i++) {
        const elem = tokenElements[i];
        let expectedToken = tokensWithWeights[i].token;
        if (expectedToken.endsWith('\n')) {
          expectedToken = expectedToken.slice(0, -1) + ' ';
          expect(elem.nextElementSibling?.classList ?? [])
              .toContain('row-break');
        }
        if (expectedToken.startsWith('\n')) {
          expectedToken = ' ' + expectedToken.slice(1);
          expect(elem.previousElementSibling?.classList ?? [])
              .toContain('row-break');
        }
        expect(elem.innerText).toEqual(expectedToken);
      }
    });

    it('should mark a selected token', async () => {
      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      for (let i = 0; i < tokensWithWeights.length; i++) {
        if (tokensWithWeights[i].selected) {
          expect(tokenElements[i]).toHaveClass('selected');
        } else {
          expect(tokenElements[i]).not.toHaveClass('selected');
        }
      }
    });

    it('should mark a pinned token', async () => {
      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      for (let i = 0; i < tokensWithWeights.length; i++) {
        if (tokensWithWeights[i].pinned) {
          expect(tokenElements[i]).toHaveClass('pinned');
        } else {
          expect(tokenElements[i]).not.toHaveClass('pinned');
        }
      }
    });

    it('should force show tooltips if requested', async () => {
      const tokenElements =
          tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
              'div.salient-token');
      expect(tokenElements.length).toEqual(tokensWithWeights.length);
      for (let i = 0; i < tokensWithWeights.length; i++) {
        expect(tokenElements[i].children[0] instanceof LitTooltip).toBeTrue();
        expect((tokenElements[i].children[0] as LitTooltip).forceShow)
            .toEqual(Boolean(tokensWithWeights[i].forceShowTooltip));
      }
    });
  });

  it('should respond to clicks', async () => {
    const clicked: boolean[] = [false, false];
    const tokensWithWeights: TokenWithWeight[] = [
      {
        token: 'hello',
        weight: 0.7,
        onClick: () => {
          clicked[0] = !clicked[0];
        }
      },
      {
        token: 'world',
        weight: 0.3,
        onClick: () => {
          clicked[1] = !clicked[1];
        }
      }
    ];

    const tokenChips = new TokenChips();
    tokenChips.tokensWithWeights = tokensWithWeights;
    document.body.appendChild(tokenChips);
    await tokenChips.updateComplete;

    const tokenElements =
        tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
            'div.salient-token');
    expect(tokenElements.length).toEqual(tokensWithWeights.length);
    for (let i = 0; i < tokensWithWeights.length; i++) {
      expect(tokenElements[i]).toHaveClass('clickable');
    }

    tokenElements[0].click();
    expect(clicked).toEqual([true, false]);

    tokenElements[1].click();
    expect(clicked).toEqual([true, true]);
  });

  it('should respond to mouseover and mouseout', async () => {
    const hovered: boolean[] = [false, false];
    const tokensWithWeights: TokenWithWeight[] = [
      {
        token: 'hello',
        weight: 0.7,
        onMouseover: () => {
          hovered[0] = true;
        },
        onMouseout: () => {
          hovered[0] = false;
        }
      },
      {
        token: 'world',
        weight: 0.3,
        onMouseover: () => {
          hovered[1] = true;
        },
        onMouseout: () => {
          hovered[1] = false;
        }
      }
    ];

    const tokenChips = new TokenChips();
    tokenChips.tokensWithWeights = tokensWithWeights;
    document.body.appendChild(tokenChips);
    await tokenChips.updateComplete;

    const tokenElements =
        tokenChips.renderRoot.querySelectorAll<HTMLDivElement>(
            'div.salient-token');
    expect(tokenElements.length).toEqual(tokensWithWeights.length);

    tokenElements[0].dispatchEvent(new Event('mouseover'));
    expect(hovered).toEqual([true, false]);
    tokenElements[0].dispatchEvent(new Event('mouseout'));
    expect(hovered).toEqual([false, false]);

    tokenElements[1].dispatchEvent(new Event('mouseover'));
    expect(hovered).toEqual([false, true]);
    tokenElements[1].dispatchEvent(new Event('mouseout'));
    expect(hovered).toEqual([false, false]);
  });
});
