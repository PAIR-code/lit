import 'jasmine';
import {LitElement} from 'lit';
import {FacetingControl} from './faceting_control';

import {LitApp} from '../core/app';
import {GroupService} from '../services/group_service';
import {AppState} from '../services/state_service';
import {mockMetadata} from '../lib/testing_utils';


describe('faceting control test', () => {
  let appState: AppState;
  let groupService: GroupService;
  let facetCtrl: FacetingControl;
  let facetButton: HTMLButtonElement;
  let configPanel: HTMLDivElement;
  let closeButton: HTMLElement;

  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    appState = app.getService(AppState);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness).
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    appState.metadata = mockMetadata;
    appState.setCurrentDataset('sst_dev');

    groupService = app.getService(GroupService);
    facetCtrl = new FacetingControl();
    // tslint:disable-next-line:no-any (to spyOn a private, readonly property)
    (facetCtrl as any).groupService = groupService;
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
    expect((facetList as HTMLSpanElement).className).toEqual('active-facets');
    expect(configPanel instanceof HTMLDivElement).toBeTrue();
    expect((configPanel as HTMLDivElement).className)
        .toEqual('config-panel popup-container');
  });

  it('shows configPanel after facet button click', async () => {
    facetButton.click();
    await facetCtrl.updateComplete;
    expect(configPanel.style.display).toEqual('block');
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
});
