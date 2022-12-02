import 'jasmine';
import {Checkbox} from '@material/mwc-checkbox';
import {LitElement} from 'lit';
import {LitCheckbox} from './checkbox';


describe('faceting control test', () => {
  let checkbox: LitCheckbox;

  beforeEach(async () => {
    // Set up.
    checkbox = new LitCheckbox();
    document.body.appendChild(checkbox);
    await checkbox.updateComplete;
  });

  afterEach(() => {
    document.body.removeChild(checkbox);
  });

  it('can be instantiated', () => {
    expect(checkbox instanceof HTMLElement).toBeTrue();
    expect(checkbox instanceof LitElement).toBeTrue();
  });

  it('comprises a div with an MWC Checkbox and a span as children', () => {
    expect(checkbox.renderRoot.children.length).toEqual(1);

    const [innerDiv] = checkbox.renderRoot.children;
    expect(innerDiv instanceof HTMLDivElement).toBeTrue();
    expect((innerDiv as HTMLDivElement).className).toEqual(' wrapper ');
    expect(innerDiv.children.length).toEqual(2);

    const [mwcCheckbox, label] = innerDiv.children;
    expect(mwcCheckbox instanceof Checkbox).toBeTrue();
    expect(label instanceof HTMLSpanElement).toBeTrue();
    expect((label as HTMLSpanElement).className).toEqual('checkbox-label');
  });

  it('toggles checked state when the box is clicked', async () => {
    expect(checkbox.checked).toBeFalse();

    const mwcCheckbox =
      checkbox.renderRoot.querySelector<Checkbox>('lit-mwc-checkbox-internal')!;
    mwcCheckbox.click();
    await checkbox.updateComplete;
    expect(checkbox.checked).toBeTrue();

    mwcCheckbox.click();
    await checkbox.updateComplete;
    expect(checkbox.checked).toBeFalse();
  });

  it('toggles checked state when the label is clicked', async () => {
    expect(checkbox.checked).toBeFalse();

    const label =
      checkbox.renderRoot.querySelector<HTMLSpanElement>('span.checkbox-label')!;
    label.click();
    await checkbox.updateComplete;
    expect(checkbox.checked).toBeTrue();

    label.click();
    await checkbox.updateComplete;
    expect(checkbox.checked).toBeFalse();
  });
});
