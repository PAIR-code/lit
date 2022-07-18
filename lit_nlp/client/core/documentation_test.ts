import 'jasmine';
import {LitElement} from 'lit';
import {DocumentationComponent} from './documentation';

import {Checkbox} from '@material/mwc-checkbox';
import {LitApp} from '../core/app';
import {LitCheckbox} from '../elements/checkbox';
import {mockMetadata} from '../lib/testing_utils';
import {AppState} from '../services/services';


describe('documentation display test', () => {
  let docComponent: DocumentationComponent;
  let prevButton: HTMLButtonElement;
  let nextButton: HTMLButtonElement;
  let docElem: HTMLElement;
  beforeEach(async () => {
    // Set up.
    const app = new LitApp();
    const appState = app.getService(AppState);
    // Stop appState from trying to make the call to the back end
    // to load the data (causes test flakiness).
    spyOn(appState, 'loadData').and.returnValue(Promise.resolve());
    appState.metadata = mockMetadata;
    appState.setCurrentDataset('sst_dev');

    // Reset the hideDialog flag in localStorage between tests
    localStorage.setItem('hideDialog', false.toString());

    docComponent = new DocumentationComponent(appState);
    document.body.appendChild(docComponent);
    await docComponent.updateComplete;

    const buttons = docComponent.renderRoot.querySelectorAll('button');
    prevButton = buttons[0];
    nextButton = buttons[1];
    docElem = docComponent.renderRoot.querySelector('#doc') as HTMLElement;
  });

  afterEach(() => {
    document.body.removeChild(docComponent);
  });

  it('can be instantiated', () => {
    expect(docComponent instanceof HTMLElement).toBeTrue();
    expect(docComponent instanceof LitElement).toBeTrue();
  });

  it('can be opened and closed', async () => {
    expect(docComponent.isOpen).toBeTrue();
    expect(docElem.classList.length).toBe(0);
    expect(docElem.style.display).toBe('');
    docComponent.close();
    await docComponent.updateComplete;
    expect(docComponent.isOpen).toBeFalse();
    expect(docElem.classList.length).toBe(1);
    expect(docElem.classList[0]).toBe('hide');
    docComponent.open();
    await docComponent.updateComplete;
    expect(docComponent.isOpen).toBeTrue();
    expect(docElem.classList.length).toBe(0);
  });

  it('can be navigated', async () => {
    expect(docComponent.isOpen).toBeTrue();
    expect(docComponent.totalPages).toBe(7);
    expect(docComponent.currentPage).toBe(0);
    nextButton.click();
    await docComponent.updateComplete;
    expect(docComponent.currentPage).toBe(1);
    prevButton.click();
    await docComponent.updateComplete;
    expect(docComponent.currentPage).toBe(0);
    for (let i = 0; i < docComponent.totalPages; i++) {
      nextButton.click();
      await docComponent.updateComplete;
    }
    expect(docComponent.isOpen).toBeFalse();
  });

  it('closes when the checkbox is checked', async () => {
    await docComponent.updateComplete;

    const litCheckbox = docComponent.renderRoot.querySelector('lit-checkbox') as LitCheckbox;
    const mwcCheckbox = litCheckbox.renderRoot.querySelector('lit-mwc-checkbox-internal') as Checkbox;
    const doNotShowCheckbox = mwcCheckbox.renderRoot.querySelector("input[type='checkbox']") as HTMLInputElement;

    doNotShowCheckbox.click();
    await docComponent.updateComplete;

    expect(doNotShowCheckbox.checked).toBeTrue();
    expect(docComponent.isOpen).toBeFalse();
  });
});
