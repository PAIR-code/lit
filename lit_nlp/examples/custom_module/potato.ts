/**
 * @fileoverview They're red, they're white, they're brown,
 * they get that way under-ground...
 *
 * This defines a custom module. The @customElement decorator registers
 * this and makes it available to the LIT framework, which can then construct it
 * from the tag name ('potato-module') when this is specified in the LIT layout
 * (see demo.py).
 */

// tslint:disable:no-new-decorators
import {customElement} from 'lit/decorators.js';
import { html} from 'lit';
import {styleMap} from 'lit/directives/style-map.js';

import {LitModule} from '../../client/core/lit_module';
import {ModelInfoMap, Spec} from '../../client/lib/types';

/** Custom LIT module. Delicious baked, mashed, or fried. */
@customElement('potato-module')
export class PotatoModule extends LitModule {
  static override title = 'Potato';
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) => {
        return html`
            <potato-module model=${model} .shouldReact=${shouldReact}
                           selectionServiceIndex=${selectionServiceIndex}>
            </potato-module>`;
      };

  override renderImpl() {
    const style = styleMap({'width': '100%', 'height': '100%'});
    // clang-format off
    return html`
      <a href="https://potato.io/">
        <img src="static/potato.svg" style=${style}>
      </a>`;
    // clang-format on
  }

  static override shouldDisplayModule(
      modelSpecs: ModelInfoMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'potato-module': PotatoModule;
  }
}
