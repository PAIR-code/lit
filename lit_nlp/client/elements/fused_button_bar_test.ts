/**
 * @license
 * Copyright 2023 Google LLC
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

import {FusedButtonBar} from './fused_button_bar';
import {LitTooltip} from './tooltip';


describe('FusedButtonBar test', () => {
  let buttonBar: FusedButtonBar;

  beforeEach(async () => {
    buttonBar = new FusedButtonBar();
    buttonBar.options = [
      {text: 'foo', tooltipText: 'Foo!', selected: false},
      {text: 'bar', tooltipText: 'Bar!', selected: false},
      {text: 'baz', tooltipText: 'Baz!', selected: true}
    ];
    document.body.appendChild(buttonBar);
    await buttonBar.updateComplete;
  });

  it('should instantiate correctly', () => {
    expect(buttonBar).toBeDefined();
    expect(buttonBar instanceof HTMLElement).toBeTrue();
    expect(buttonBar instanceof LitElement).toBeTrue();
  });

  it('should render an optional label', async () => {
    let elements =
        buttonBar.renderRoot.querySelectorAll<HTMLDivElement>('div.label');
    expect(elements.length).toEqual(0);

    buttonBar.label = 'Button bar';
    buttonBar.requestUpdate();
    await buttonBar.updateComplete;

    elements =
        buttonBar.renderRoot.querySelectorAll<HTMLDivElement>('div.label');
    expect(elements.length).toEqual(1);
    expect(elements[0].innerText).toEqual(buttonBar.label);
  });

  it('should render a set of elements', async () => {
    const elements = buttonBar.renderRoot.querySelectorAll<HTMLDivElement>(
        'div.button-bar-item');
    expect(elements.length).toEqual(buttonBar.options.length);
    expect(elements[0].innerText).toEqual(buttonBar.options[0].text);
    expect(elements[0].children[0]).toBeInstanceOf(LitTooltip);
    expect(elements[0].querySelector<HTMLButtonElement>('button'))
        .toBeInstanceOf(HTMLButtonElement);
  });

  it('should mark a selected token', async () => {
    const elements = buttonBar.renderRoot.querySelectorAll<HTMLDivElement>(
        'div.button-bar-item');
    expect(elements.length).toEqual(buttonBar.options.length);
    for (let i = 0; i < buttonBar.options.length; i++) {
      const button = elements[i].querySelector<HTMLButtonElement>('button');
      if (buttonBar.options[i].selected) {
        expect(button).toHaveClass('active');
      } else {
        expect(button).not.toHaveClass('active');
      }
    }
  });

  it('should have disabled buttons if not clickable', async () => {
    const buttons =
        buttonBar.renderRoot.querySelectorAll<HTMLButtonElement>('button');
    expect(buttons.length).toEqual(buttonBar.options.length);
    for (let i = 0; i < buttonBar.options.length; i++) {
      expect(buttons[i].disabled).toBeTrue();
    }
  });

  it('should respond to clicks', async () => {
    for (let i = 0; i < buttonBar.options.length; i++) {
      buttonBar.options[i].onClick = async () => {
        buttonBar.options[i].selected = !buttonBar.options[i].selected;
        buttonBar.requestUpdate();
        await buttonBar.updateComplete;
      };
    }
    buttonBar.requestUpdate();
    await buttonBar.updateComplete;

    const buttons =
        buttonBar.renderRoot.querySelectorAll<HTMLButtonElement>('button');
    expect(buttons.length).toEqual(buttonBar.options.length);
    for (let i = 0; i < buttonBar.options.length; i++) {
      // Not disabled since onClick is set.
      expect(buttons[i].disabled).toBeFalse();
    }

    buttons[0].click();
    expect(buttonBar.options[0].selected).toBeTrue();

    buttons[0].click();
    expect(buttonBar.options[0].selected).toBeFalse();
  });
});