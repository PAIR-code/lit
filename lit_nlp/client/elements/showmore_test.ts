import 'jasmine';

import {html, render} from 'lit';
import {LitElement} from 'lit';

import {LitShowMore} from './showmore';

describe('showmore test', () => {
  let showmore: LitShowMore;
  const CHANGE_HANDLERS = {
    showmore: (e: Event) => {},
  };

  beforeEach(async () => {
    showmore = new LitShowMore();
    document.body.appendChild(showmore);
    await showmore.updateComplete;
  });

  it('should instantiate correctly', () => {
    expect(showmore).toBeDefined();
    expect(showmore).toBeInstanceOf(HTMLElement);
    expect(showmore).toBeInstanceOf(LitElement);
  });

  it('is initially visible', async () => {
    const content = html`
    <lit-showmore
      id="visible"
      @show-more=${CHANGE_HANDLERS.showmore}>
    </lit-showmore>`;
    render(content, document.body);
    const queryString = 'lit-showmore#visible';
    const showMoreIcon =
      document.body.querySelector<LitShowMore>(queryString)!;
    expect(showMoreIcon).toBeDefined();
    await showMoreIcon.updateComplete;
    const contentDiv =
        showMoreIcon.renderRoot.children[0] as HTMLDivElement;
    expect(contentDiv).toBeDefined();
    expect(contentDiv.innerHTML).toContain('more_horiz');
  });

  it('emits an event when clicked', async() => {
    const spy = spyOn(CHANGE_HANDLERS, 'showmore');
    const content = html`
    <lit-showmore
      id="event-when-clicked"
      @showmore=${CHANGE_HANDLERS.showmore}>
    </lit-showmore>`;
    render(content, document.body);
    const queryString = 'lit-showmore#event-when-clicked';
    const showMoreIcon =
      document.body.querySelector<LitShowMore>(queryString)!;
    await showMoreIcon.updateComplete;
    const icon = showMoreIcon.renderRoot.querySelector(
      'span > lit-tooltip > span[slot="tooltip-anchor"]'
    ) as HTMLSpanElement;
    icon.click();
    await showMoreIcon.updateComplete;

    expect(spy).toHaveBeenCalled();
  });
});