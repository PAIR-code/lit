/**
 * @fileoverview A reusable expansion panel element for LIT
 *
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
import {html, render, LitElement} from 'lit';
import {ExpansionPanel, ExpansionToggle} from './expansion_panel';

describe('expansion panel test', () => {
  let expansionPanel: ExpansionPanel;

  const expansionHandler = (event: Event) => {
    const customEvent = event as CustomEvent<ExpansionToggle>;
    expect(customEvent.detail.isExpanded).toBeDefined();
  };

  beforeEach(async () => {
    expansionPanel = new ExpansionPanel();
    document.body.appendChild(expansionPanel);
    document.body.addEventListener('expansion-toggle', expansionHandler);
    await expansionPanel.updateComplete;
  });

  afterEach(() => {
    document.body.removeEventListener('expansion-toggle', expansionHandler);
  });

  it('should instantiate correctly', () => {
    expect(expansionPanel).toBeDefined();
    expect(expansionPanel instanceof HTMLElement).toBeTrue();
    expect(expansionPanel instanceof LitElement).toBeTrue();
  });

  it('is initially collapsed', () => {
    expect(expansionPanel.renderRoot.children.length).toEqual(1);
    const [firstChild] = expansionPanel.renderRoot.children;
    expect(firstChild instanceof HTMLDivElement).toBeTrue();
    expect((firstChild as HTMLDivElement).className)
        .toEqual('expansion-header');
  });

  it('expands when you click the header and emits an event', async () => {
    const [firstChild] = expansionPanel.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await expansionPanel.updateComplete;

    expect(expansionPanel.renderRoot.children.length).toEqual(2);
    const [, secondChild] = expansionPanel.renderRoot.children;
    expect(secondChild instanceof HTMLDivElement).toBeTrue();
    expect((secondChild as HTMLDivElement).className)
        .toEqual(' expansion-content ');
  });

  it('collapses when you click the header a second time and emits an event',
      async () => {
        const [firstChild] = expansionPanel.renderRoot.children;
        (firstChild as HTMLDivElement).click();
        await expansionPanel.updateComplete;

        (firstChild as HTMLDivElement).click();
        await expansionPanel.updateComplete;

        expect(expansionPanel.renderRoot.children.length).toEqual(1);
      });

  it('respects the expanded flag', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'} expanded>
        </expansion-panel>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    const [, content] = panel.renderRoot.children;
    expect(panel.renderRoot.children.length).toEqual(2);
    expect((content as HTMLElement).className).toEqual(' expansion-content ');
  });

  it('respects the padLeft flag', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'} expanded padLeft>
        </expansion-panel>`;
    await render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    const [, content] = panel.renderRoot.children;
    expect(panel.renderRoot.children.length).toEqual(2);
    expect((content as HTMLElement).style.width).toEqual('calc(100% - 8px)');
    expect((content as HTMLElement).className)
        .toEqual(' expansion-content pad-left ');
  });

  it('respects the padRight flag', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'} expanded padRight>
        </expansion-panel>`;
    await render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;


    const [, content] = panel.renderRoot.children;
    expect(panel.renderRoot.children.length).toEqual(2);
    expect((content as HTMLElement).style.width).toEqual('calc(100% - 16px)');
    expect((content as HTMLElement).className)
        .toEqual(' expansion-content pad-right ');
  });

  it('respects padLeft and padRight simultaneously', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'}
                          expanded padLeft padRight>
        </expansion-panel>`;
    await render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;


    const [, content] = panel.renderRoot.children;
    expect(panel.renderRoot.children.length).toEqual(2);
    expect((content as HTMLElement).style.width).toEqual('calc(100% - 24px)');
    expect((content as HTMLElement).className)
        .toEqual(' expansion-content pad-left pad-right ');
  });

  it('should render a single Element in its slot from a template', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'}>
          <div>This is a test div</div>
        </expansion-panel>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    const [firstChild] = panel.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await panel.updateComplete;

    const slot =
        panel.shadowRoot!.querySelector('slot:not([name])') as HTMLSlotElement;
    const slottedNodes = slot!.assignedNodes({flatten: true})
                              .filter(n => n instanceof HTMLDivElement);

    expect(slottedNodes.length).toEqual(1);
    expect(slottedNodes[0] instanceof HTMLDivElement).toBeTrue();
  });

  it('should render many Elements in its slot from a template', async () => {
    const template = html`
        <expansion-panel .label=${'test expansion panel'}>
          <div>This is a test div</div>
          <div>This is another test div</div>
          <div>This is a third test div</div>
        </expansion-panel>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('expansion-panel');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    const [firstChild] = panel.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await panel.updateComplete;

    const slot =
        panel.shadowRoot!.querySelector('slot:not([name])') as HTMLSlotElement;
    const slottedNodes = slot!.assignedNodes({flatten: true})
                              .filter(n => n instanceof HTMLDivElement);

    expect(slottedNodes.length).toEqual(3);
    for (const node of slottedNodes) {
      expect(node instanceof HTMLDivElement).toBeTrue();
    }
  });

  it('should render bar content from a named slot from a template',
     async () => {
       const template = html`
        <expansion-panel .label=${'test expansion panel'}>
          <div slot="bar-content">This goes in the bar</div>
          <div>This is a test div</div>
        </expansion-panel>`;
       render(template, document.body);
       const panels = document.body.querySelectorAll('expansion-panel');
       const panel = panels[panels.length - 1];
       await panel.updateComplete;

       const [firstChild] = panel.renderRoot.children;
       (firstChild as HTMLDivElement).click();
       await panel.updateComplete;

       const namedSlot = panel.shadowRoot!.querySelector(
                             'slot[name=bar-content]') as HTMLSlotElement;
       const namedSlotNodes = namedSlot.assignedNodes({flatten: true})
                                  .filter(n => n instanceof HTMLDivElement);

       expect(namedSlotNodes.length).toEqual(1);
       expect(namedSlotNodes[0] instanceof HTMLDivElement).toBeTrue();

       // Check the non-named slot is still handled correctly.
       const slot = panel.shadowRoot!.querySelector('slot:not([name])') as
           HTMLSlotElement;
       const slottedNodes = slot.assignedNodes({flatten: true})
                                .filter(n => n instanceof HTMLDivElement);

       expect(slottedNodes.length).toEqual(1);
       expect(slottedNodes[0] instanceof HTMLDivElement).toBeTrue();
     });
});
