import 'jasmine';
import {LitElement} from 'lit';
import {FacetingControl, FacetsChange} from './faceting_control';

import {Checkbox} from '@material/mwc-checkbox';
import {LitApp} from '../core/app';
import {LitCheckbox} from '../elements/checkbox';
import {mockMetadata} from '../lib/testing_utils';
import {AppState, DataService, GroupService} from '../services/services';


describe('faceting control test', () => {
  let facetCtrl: FacetingControl;
  let facetButton: HTMLButtonElement;
  let configPanel: HTMLDivElement;
  let closeButton: HTMLElement;

  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    const appState = app.getService(AppState);
    const dataService = app.getService(DataService);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness).
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    appState.metadata = mockMetadata;
    appState.setCurrentDataset('sst_dev');

    const groupService = new GroupService(appState, dataService);
    facetCtrl = new FacetingControl(groupService);
    document.body.appendChild(facetCtrl);
    await facetCtrl.updateComplete;

    facetButton =
        facetCtrl.renderRoot.children[0].children[0] as HTMLButtonElement;
    configPanel =
        facetCtrl.renderRoot.children[0].children[2] as HTMLDivElement;
    closeButton =
        configPanel.querySelector('.icon-button') as HTMLElement;
  });

  afterEach(() => {
    document.body.removeChild(facetCtrl);
  });

  it('can be instantiated', () => {
    expect(facetCtrl instanceof HTMLElement).toBeTrue();
    expect(facetCtrl instanceof LitElement).toBeTrue();
  });

  it('comprises a div with a button, span, and div as children', () => {
    expect(facetCtrl.renderRoot.children.length).toEqual(1);

    const innerDiv = facetCtrl.renderRoot.children[0];
    expect(innerDiv instanceof HTMLDivElement).toBeTrue();
    expect((innerDiv as HTMLDivElement).className).toEqual('faceting-info');
    expect(innerDiv.children.length).toEqual(3);

    const [facetButton, facetList, configPanel] = innerDiv.children;
    expect(facetButton instanceof HTMLButtonElement).toBeTrue();
    expect(facetList instanceof HTMLSpanElement).toBeTrue();
    expect((facetList as HTMLSpanElement).className).toEqual(' active-facets ');
    expect(configPanel instanceof HTMLDivElement).toBeTrue();
    expect((configPanel as HTMLDivElement).className)
        .toEqual('config-panel popup-container');
  });

  it('shows configPanel after facet button click', async () => {
    facetButton.click();
    await facetCtrl.updateComplete;
    expect(configPanel.style.display).toEqual('flex');
    expect(configPanel.style.visibility).toEqual('visible');
  });

  it('hides configPanel after second facet button click', async () => {
    facetButton.click();
    await facetCtrl.updateComplete;
    facetButton.click();
    await facetCtrl.updateComplete;
    expect(configPanel.style.display).toEqual('none');
    expect(configPanel.style.visibility).toEqual('hidden');
  });

  it('hides configPanel after closeButton click', async () => {
    facetButton.click();
    await facetCtrl.updateComplete;
    closeButton.click();
    await facetCtrl.updateComplete;
    expect(configPanel.style.display).toEqual('none');
    expect(configPanel.style.visibility).toEqual('hidden');
  });

  it('hides configPanel by clicking anything else', async () => {
    facetButton.click();
    await facetCtrl.updateComplete;
    document.body.click();
    await facetCtrl.updateComplete;
    expect(configPanel.style.display).toEqual('none');
    expect(configPanel.style.visibility).toEqual('hidden');
  });

  it('emits a custom facets-change event after checkbox click', async () => {
    const facetChangeHandler = (event: Event) => {
      const customEvent = event as CustomEvent<FacetsChange>;
      expect(customEvent.detail.features.length).toBe(1);
      expect(customEvent.detail.features[0]).toBe('label');
      expect(customEvent.detail.bins).toBeTruthy();
    };
    document.body.addEventListener('facets-change', facetChangeHandler);
    facetButton.click();
    await facetCtrl.updateComplete;

    const featureRow = configPanel.querySelector('div.feature-options-row') as HTMLDivElement;
    const litCheckbox = featureRow.querySelector('lit-checkbox') as LitCheckbox;
    const mwcCheckbox = litCheckbox.renderRoot.querySelector('lit-mwc-checkbox-internal') as Checkbox;
    const input = mwcCheckbox.renderRoot.querySelector("input[type='checkbox']") as HTMLInputElement;
    expect(input.checked).toBeFalse();
    input.click();
    await facetCtrl.updateComplete;
    expect(input.checked).toBeTrue();
    document.body.removeEventListener('facets-change', facetChangeHandler);
  });

  it('can be reset programmatically', async () => {
    let expectedFacetCount = 1;
    const facetChangeHandler = (event: Event) => {
      const customEvent = event as CustomEvent<FacetsChange>;
      expect(customEvent.detail.features.length).toBe(expectedFacetCount);
      expect(customEvent.detail.bins).toBeTruthy();
    };

    document.body.addEventListener('facets-change', facetChangeHandler);
    facetButton.click();
    await facetCtrl.updateComplete;

    const featureRow = configPanel.querySelector('div.feature-options-row') as HTMLDivElement;
    const litCheckbox = featureRow.querySelector('lit-checkbox') as LitCheckbox;
    const mwcCheckbox = litCheckbox.renderRoot.querySelector('lit-mwc-checkbox-internal') as Checkbox;
    const input = mwcCheckbox.renderRoot.querySelector("input[type='checkbox']") as HTMLInputElement;
    input.click();
    await facetCtrl.updateComplete;
    expectedFacetCount = 0;
    facetCtrl.reset();
    await facetCtrl.updateComplete;
    document.body.removeEventListener('facets-change', facetChangeHandler);
  });
});
