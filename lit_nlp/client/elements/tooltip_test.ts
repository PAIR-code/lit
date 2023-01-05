import 'jasmine';

import {LitElement} from 'lit';

import {LitTooltip} from './tooltip';

describe('tooltip test', () => {
  let tooltip: LitTooltip;
  let tooltipText: HTMLSpanElement;

  beforeEach(async () => {
    tooltip = new LitTooltip();
    tooltip.content = 'Test content';
    document.body.appendChild(tooltip);
    await tooltip.updateComplete;

    tooltipText =
        tooltip.renderRoot.querySelector<HTMLSpanElement>('span.tooltip-text')!;
  });

  it('should instantiate correctly', () => {
    expect(tooltip).toBeDefined();
    expect(tooltip instanceof HTMLElement).toBeTrue();
    expect(tooltip instanceof LitElement).toBeTrue();
  });

  it('is initially hidden with correct contents', async () => {
    expect(tooltipText).toBeDefined();
    expect(tooltipText.innerHTML).toContain('Test content');
    const style = window.getComputedStyle(tooltipText);
    expect(style.getPropertyValue('visibility')).toEqual('hidden');
    expect(tooltipText).not.toHaveClass('above');
  });

  it('conditionally renders aria title', async () => {
    expect(tooltip.renderAriaTitle()).not.toEqual(``);
    tooltip.shouldRenderAriaTitle = false;
    expect(tooltip.renderAriaTitle()).toEqual(``);
  });

  it('conditionally updates tooltip position', async () => {
    tooltip.tooltipPosition = 'above';
    await tooltip.updateComplete;
    expect(tooltipText).toHaveClass('above');
  });

  it('toggles visibility on click', async () => {
    // Click to make visible.
    const tooltipIcon =
        tooltip.renderRoot.querySelector<HTMLElement>('.lit-tooltip')!;
    tooltipIcon.click();
    await tooltip.updateComplete;

    const style = window.getComputedStyle(tooltipText);
    expect(style.getPropertyValue('visibility')).toEqual('visible');

    // Click to hide.
    tooltipIcon.click();
    await tooltip.updateComplete;
    expect(style.getPropertyValue('visibility')).toEqual('hidden');
  });
});
