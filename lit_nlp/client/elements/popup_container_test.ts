import 'jasmine';

import {html, LitElement, render} from 'lit';

import {PopupContainer} from './popup_container';

describe('popup container test', () => {
  let popupContainer: PopupContainer;

  beforeEach(async () => {
    popupContainer = new PopupContainer();
    document.body.appendChild(popupContainer);
    await popupContainer.updateComplete;
  });

  it('should instantiate correctly', () => {
    expect(popupContainer).toBeDefined();
    expect(popupContainer instanceof HTMLElement).toBeTrue();
    expect(popupContainer instanceof LitElement).toBeTrue();
  });

  it('is initially collapsed', () => {
    expect(popupContainer.renderRoot.children.length).toEqual(1);
    const [firstChild] = popupContainer.renderRoot.children;
    expect(firstChild instanceof HTMLDivElement).toBeTrue();
    expect((firstChild as HTMLDivElement).className)
        .toEqual('popup-toggle-anchor');
  });

  it('expands when you click the toggle', async () => {
    const [firstChild] = popupContainer.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await popupContainer.updateComplete;

    expect(popupContainer.renderRoot.children.length).toEqual(2);
    const [, secondChild] = popupContainer.renderRoot.children;
    expect(secondChild instanceof HTMLDivElement).toBeTrue();
    expect((secondChild as HTMLDivElement).className)
        .toEqual('popup-outer-holder');
  });

  it('collapses when you click the header a second time', async () => {
    const [firstChild] = popupContainer.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await popupContainer.updateComplete;

    (firstChild as HTMLDivElement).click();
    await popupContainer.updateComplete;

    expect(popupContainer.renderRoot.children.length).toEqual(1);
  });

  it('collapses when you click anything else', async () => {
    const [firstChild] = popupContainer.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await popupContainer.updateComplete;

    document.body.click();
    await popupContainer.updateComplete;

    expect(popupContainer.renderRoot.children.length).toEqual(1);
  });

  it('respects the expanded flag', async () => {
    const template = html`<popup-container expanded></popup-container>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('popup-container');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    const [, content] = panel.renderRoot.children;
    expect(panel.renderRoot.children.length).toEqual(2);
    expect((content as HTMLElement).className).toEqual('popup-outer-holder');
  });

  it('should render a single Element in its slot from a template', async () => {
    const template = html`
        <popup-container>
          <div>This is a test div</div>
        </popup-container>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('popup-container');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    // Click to expand
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
        <popup-container>
          <div>This is a test div</div>
          <div>This is another test div</div>
          <div>This is a third test div</div>
        </popup-container>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('popup-container');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    // Click to expand
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

  it('should render controls from a named slot from a template', async () => {
    const template = html`
        <popup-container>
          <button slot="toggle-anchor">Open panel</button>
          <div>This is a test div</div>
        </popup-container>`;
    render(template, document.body);
    const panels = document.body.querySelectorAll('popup-container');
    const panel = panels[panels.length - 1];
    await panel.updateComplete;

    // Click to expand
    const [firstChild] = panel.renderRoot.children;
    (firstChild as HTMLDivElement).click();
    await panel.updateComplete;

    const namedSlot = panel.shadowRoot!.querySelector(
                          'slot[name=toggle-anchor]') as HTMLSlotElement;
    const namedSlotNodes = namedSlot.assignedNodes({flatten: true})
                               .filter(n => n instanceof HTMLButtonElement);

    expect(namedSlotNodes.length).toEqual(1);
    expect(namedSlotNodes[0] instanceof HTMLButtonElement).toBeTrue();

    // Check the non-named slot is still handled correctly.
    const slot =
        panel.shadowRoot!.querySelector('slot:not([name])') as HTMLSlotElement;
    const slottedNodes = slot.assignedNodes({flatten: true})
                             .filter(n => n instanceof HTMLDivElement);

    expect(slottedNodes.length).toEqual(1);
    expect(slottedNodes[0] instanceof HTMLDivElement).toBeTrue();
  });
});
