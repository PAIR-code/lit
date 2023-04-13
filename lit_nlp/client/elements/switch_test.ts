import 'jasmine';

import {Switch} from '@material/mwc-switch/mwc-switch';
import {LitElement} from 'lit';

import {LitSwitch} from './switch';


describe('switch test', () => {
  let litSwitch: LitSwitch;

  beforeEach(async () => {
    // Set up.
    litSwitch = new LitSwitch();
    document.body.appendChild(litSwitch);
    await litSwitch.updateComplete;
  });

  afterEach(() => {
    document.body.removeChild(litSwitch);
  });

  it('can be instantiated', () => {
    expect(litSwitch instanceof HTMLElement).toBeTrue();
    expect(litSwitch instanceof LitElement).toBeTrue();
  });

  it('comprises a div with an MWC Switch and two divs as children', () => {
    expect(litSwitch.renderRoot.children.length).toEqual(1);

    const [innerDiv] = litSwitch.renderRoot.children;
    expect(innerDiv instanceof HTMLDivElement).toBeTrue();
    expect((innerDiv as HTMLDivElement).className)
        .toContain('switch-container');
    expect(innerDiv.children.length).toEqual(3);

    const [labelLeft, mwcSwitch, labelRight] = innerDiv.children;
    expect(labelLeft instanceof HTMLDivElement).toBeTrue();
    expect((labelLeft as HTMLDivElement).className)
        .toEqual('switch-label label-left');
    expect(mwcSwitch instanceof Switch).toBeTrue();
    expect(labelRight instanceof HTMLDivElement).toBeTrue();
    expect((labelRight as HTMLDivElement).className)
        .toEqual('switch-label label-right');
  });

  it('toggles state when the element is clicked', async () => {
    expect(litSwitch.selected).toBeFalse();

    const [innerDiv] = litSwitch.renderRoot.children;
    (innerDiv as HTMLDivElement).click();
    await litSwitch.updateComplete;
    expect(litSwitch.selected).toBeTrue();

    (innerDiv as HTMLDivElement).click();
    await litSwitch.updateComplete;
    expect(litSwitch.selected).toBeFalse();
  });
});
