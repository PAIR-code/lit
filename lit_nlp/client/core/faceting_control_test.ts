import 'jasmine';

import {Checkbox} from '@material/mwc-checkbox';
import {LitElement} from 'lit';

import {LitApp} from '../core/app';
import {LitCheckbox} from '../elements/checkbox';
import {PopupContainer} from '../elements/popup_container';
import {mockMetadata} from '../lib/testing_utils';
import {AppState, DataService, GroupService} from '../services/services';

import {FacetingControl, FacetsChange} from './faceting_control';


describe('faceting control test', () => {
  let facetCtrl: FacetingControl;
  let popupContainer: PopupContainer;
  let facetButton: HTMLButtonElement;
  let configPanel: HTMLDivElement;

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

    popupContainer = facetCtrl.renderRoot.children[0] as PopupContainer;
    facetButton = popupContainer.children[0].children[0] as HTMLButtonElement;
    configPanel = popupContainer.children[1] as HTMLDivElement;
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

    expect(popupContainer instanceof PopupContainer).toBeTrue();
    expect(popupContainer.children.length).toEqual(2);

    const [facetingInfo, configPanel] = popupContainer.children;
    expect(facetingInfo instanceof HTMLDivElement).toBeTrue();
    expect((facetingInfo as HTMLDivElement).className).toEqual('faceting-info');
    const [facetButton, facetList] = facetingInfo.children;
    expect(facetButton instanceof HTMLButtonElement).toBeTrue();
    expect(facetList instanceof HTMLDivElement).toBeTrue();
    expect((facetList as HTMLDivElement).className).toEqual(' active-facets ');
    expect(configPanel instanceof HTMLDivElement).toBeTrue();
    expect((configPanel as HTMLDivElement).className).toEqual('config-panel');
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
