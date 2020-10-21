/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Client-side (UI) code for the LIT tool.
 */

import {customElement, html, LitElement, property} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import '@material/mwc-icon';

import {LitRenderConfig, RenderConfig} from '../services/modules_service';
import {ModulesService} from '../services/services';

import {app} from './lit_app';
import {LitModule} from './lit_module';
import {styles} from './modules.css';
import {LitWidget} from './widget_group';


/**
 * The component responsible for rendering the selected and available lit
 * modules. This component does not extend from MobxLitElement, as we want
 * to explicitly control when it rerenders (via the setRenderModulesCallback).
 */
@customElement('lit-modules')
export class LitModules extends LitElement {
  private readonly modulesService = app.getService(ModulesService);
  @property({type: Number})
  mainSectionHeight = this.modulesService.getSetting('mainHeight') || 45;

  static get styles() {
    return styles;
  }

  firstUpdated() {
    // We set up a callback in the modulesService to allow it to explicitly
    // trigger a rerender of this component when visible modules have been
    // updated by the user. Normally we'd do this in a reactive way, but we'd
    // like as fine-grain control over layout rendering as possible.
    this.modulesService.setRenderModulesCallback(() => {
      this.requestUpdate();
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    // We clear the callback if / when the lit-modules component is removed.
    this.modulesService.setRenderModulesCallback(() => {});
  }

  connectedCallback() {
    super.connectedCallback();
    this.style.setProperty(
        'align-self',
        this.modulesService.getSetting('centerPage') ? `center` : `unset`);
  }

  updated() {
    // Since the widget parent element is responsible for displaying the load
    // status of its child litModule, and we can't provide a callback to a
    // dynamically created template string component (in
    // this.renderModuleWidget), we need to imperatively set a callback on the
    // child litModule to allow it to set the loading state of its parent
    // widget.
    const widgetElements = this.shadowRoot!.querySelectorAll('lit-widget');
    widgetElements.forEach(widgetElement => {
      // Currently, a widget can only contain one LitModule, so we just select
      //  the first child and ensure it's a LitModule.
      const litModuleElement = widgetElement.children[0];
      if (litModuleElement instanceof LitModule) {
        litModuleElement.setIsLoading = (isLoading: boolean) => {
          (widgetElement as LitWidget).isLoading = isLoading;
        };
      }
    });
  }

  render() {
    const layout = this.modulesService.getRenderLayout();
    const mainPanelConfig = layout['Main'];
    const compGroupNames = Object.keys(layout).filter(k => k !== 'Main');

    // Initially set the selected tab to the first tab.
    if (this.modulesService.selectedTab === '') {
      this.modulesService.selectedTab = compGroupNames[0];
    }

    const styles = styleMap({'height': `${this.mainSectionHeight}vh`});

    // clang-format off
    return html`
      <div id='main-panel' style=${styles}>
        ${this.renderMainPanel(mainPanelConfig)}
      </div>
      <div id='center-bar'>
        <div id='tabs'>
          ${this.renderTabs(compGroupNames)}
        </div>
        <div id='drag-container'>
          <mwc-icon class="drag-icon">menu</mwc-icon>
          <div id='drag-handler' draggable='true'
              @drag=${(e: DragEvent) => {this.onBarDragged(e);}}>
          </div>
        </div>
      </div>
      <div id='component-groups'>
        ${this.renderComponentGroups(layout, compGroupNames)}
      </div>
    `;
    // clang-format on
  }

  private onBarDragged(e: DragEvent) {
    const main = this.shadowRoot!.getElementById('main-panel')!;
    const mainTopPos = main.getBoundingClientRect().top;
    this.mainSectionHeight =
        Math.floor((e.clientY - mainTopPos - 10) / window.innerHeight * 100);
  }

  /**
   * Render the tabbed groups of components.
   * @param compGroupNames Names of the components to render
   */
  renderComponentGroups(layout: LitRenderConfig, compGroupNames: string[]) {
    return compGroupNames.map((compGroupName) => {
      const configs = layout[compGroupName];
      const componentsHTML =
          configs.map(configGroup => 
            html`
            <lit-widget-group .configGroup=${configGroup}></lit-widget-group>`);
      const selected = this.modulesService.selectedTab === compGroupName;
      const classes = classMap({selected, 'components-group-holder': true});
      return html`
        <div class=${classes}>
          ${componentsHTML}
        </div>`;
    });
  }


  /**
   * Render the tabs of the selection groups at the bottom of the layout.
   * @param compGroupNames Names of the tabs to render
   */
  renderTabs(compGroupNames: string[]) {
    return compGroupNames.map((compGroupName) => {
      const name = compGroupName;
      const onclick = (e: Event) => {
        this.modulesService.selectedTab = name;
        e.preventDefault();
        // Need to trigger a manual update, since this class does not
        // respond automatically to mobx observables.
        this.requestUpdate();
      };
      const selected = this.modulesService.selectedTab === compGroupName;
      const classes = classMap({selected, tab: true});
      return html`<div class=${classes} @click=${onclick}>${
          compGroupName}</div>`;
    });
  }

  renderMainPanel(configs: RenderConfig[][]) {
    return configs.map(configGroup =>html`<lit-widget-group .configGroup=${configGroup}></lit-widget-group>`);
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-modules': LitModules;
  }
}
