import 'jasmine';
import {LitElement} from 'lit';
import {html, render} from 'lit';

import {LitTableTextCell} from './table_text_cell';

describe('table text cell test', () => {
    let textcell: LitTableTextCell;
    const CHANGE_HANDLERS = {
        showmore: (e: Event) => {},
    };

    beforeEach(async() => {
        textcell = new LitTableTextCell();
        document.body.appendChild(textcell);
        await textcell.updateComplete;
    });

    it('should instantiate correctly', () => {
        expect(textcell).toBeDefined();
        expect(textcell).toBeInstanceOf(HTMLElement);
        expect(textcell).toBeInstanceOf(LitElement);
    });

    it('does not display show more on short text', async () => {
        const content = "short";
        const maxWidth = 600;
        const textCellHtml = html`
        <lit-table-text-cell
          id=${content}
          .content=${content}
          .maxWidth=${maxWidth}>
        </lit-table-text-cell>`;
        render(textCellHtml, document.body);
        const queryString = `lit-table-text-cell#short`;
        const tableTextCell =
            document.body.querySelector<LitTableTextCell>(queryString)!;
        expect(tableTextCell).toBeDefined();
        await tableTextCell.updateComplete;
        const contentDiv =
            tableTextCell.renderRoot.children[0] as HTMLDivElement;
        expect(contentDiv.innerHTML).not.toContain('lit-showmore');
        expect(contentDiv.innerText).toContain('short');
    });

    it('displays show more on long text, removes icon on click', async() => {
        const content =
            "this is a long text string that should render a show more";
        const maxWidth = 20;
        const textCellHtml = html`
        <lit-table-text-cell
          id=${"long"}
          .content=${content}
          .maxWidth=${maxWidth}
          @showmore=${CHANGE_HANDLERS.showmore}>
        </lit-table-text-cell>`;
        render(textCellHtml, document.body);
        const queryString = `lit-table-text-cell#long`;
        const tableTextCell =
            document.body.querySelector<LitTableTextCell>(queryString)!;
        expect(tableTextCell).toBeDefined();
        await tableTextCell.updateComplete;
        const contentDiv =
            tableTextCell.renderRoot.children[0] as HTMLDivElement;
        expect(contentDiv.innerHTML).toContain('lit-showmore');
        expect(contentDiv.innerText).toContain('this');
        expect(contentDiv.innerText).not.toContain(content);

        const showMoreDiv = contentDiv.children[0] as HTMLSpanElement;
        showMoreDiv.dispatchEvent(new Event('showmore'));

        await tableTextCell.updateComplete;
        expect(contentDiv.innerText).toContain(content);
        expect(contentDiv.innerHTML).not.toContain('lit-showmore');
    });

});