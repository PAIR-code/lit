/**
 * @fileoverview They're red, they're white, they're brown,
 * they get that way under-ground...
 */

// tslint:disable:no-new-decorators
import {LitModule} from '../../client/core/lit_module';
import {ModelsMap, Spec} from '../../client/lib/types';
import {customElement, html} from 'lit-element';
import {styleMap} from 'lit-html/directives/style-map';

/** Custom LIT module. */
@customElement('potato-module')
export class PotatoModule extends LitModule {
  static title = 'Potato';
  static numCols = 4;
  static template = () => {
    return html`<potato-module></potato-module>`;
  };

  render() {
    const style = styleMap({'width': '100%', 'height': '100%'});
    // clang-format off
    return html`
      <a href="https://potato.io/">
        <img src="static/potato.svg" style=${style}>
      </a>`;
    // clang-format on
  }

  static shouldDisplayModule(modelSpecs: ModelsMap, datasetSpec: Spec) {
    return true;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'potato-module': PotatoModule;
  }
}
