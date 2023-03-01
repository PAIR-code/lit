import 'jasmine';

import {LitElement} from 'lit';

import {LitShowMore} from './showmore';

describe('showmore test', () => {
  let showmore: LitShowMore;

  beforeEach(async () => {
    showmore = new LitShowMore();
    showmore.visible = false;
    document.body.appendChild(showmore);
    await showmore.updateComplete;
  });

  it('should instantiate correctly', () => {
    expect(showmore).toBeDefined();
    expect(showmore instanceof HTMLElement).toBeTrue();
    expect(showmore instanceof LitElement).toBeTrue();
  });
});