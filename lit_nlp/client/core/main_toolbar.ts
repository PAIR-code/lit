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
 * LIT App selection controls toolbar
 */

import '../elements/popup_container';
import '@material/mwc-icon';

// tslint:disable:no-new-decorators
import {MobxLitElement} from '@adobe/lit-mobx';
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {classMap} from 'lit/directives/class-map';
import {styleMap} from 'lit/directives/style-map';
import {computed} from 'mobx';

import {MenuItem} from '../elements/menu';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {compareArrays, flatten, randInt} from '../lib/utils';
import {AppState, ColorService, SelectionService, SliceService} from '../services/services';

import {app} from './app';
import {styles} from './main_toolbar.css';


/**
 * The selection controls toolbar
 */
@customElement('lit-main-menu')
export class LitMainMenu extends MobxLitElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly selectionService = app.getService(SelectionService);
  private readonly sliceService = app.getService(SliceService);
  private readonly colorService = app.getService(ColorService);

  private findAllParents() {
    // This may contain duplicates, but it's fine because selectionService
    // will de-duplicate automatically.
    // Note that ancestry includes the current point.
    const allAncestors = flatten(this.selectionService.selectedIds.map(
        id => this.appState.getAncestry(id)));
    return allAncestors;
  }

  private findAllChildren() {
    const ids = new Set<string>(this.selectionService.selectedIds);
    // Check all datapoints to see if they have an ancestor in the selected
    // list. If so, add them to the list.
    for (const d of this.appState.currentInputData) {
      for (const ancestorId of this.appState.getAncestry(d.id)) {
        if (this.selectionService.isIdSelected(ancestorId)) {
          ids.add(d.id);
          continue;
        }
      }
    }
    return [...ids];
  }

  private selectRelated() {
    // selectionService will de-duplicate automatically.
    this.selectionService.selectIds(
        [...this.findAllParents(), ...this.findAllChildren()]);
  }

  // Returns items for the select menu.
  getSelectItems() {
    const selectRandomDatapoint = () => {
      const nextIndex = randInt(0, this.appState.currentInputData.length);
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    const selectAllRelatedCallback = () => {
      this.selectRelated();
    };
    const selectAllParentsCallback = () => {
      this.selectionService.selectIds([...this.findAllParents()]);
    };
    const selectAllChildrenCallback = () => {
      this.selectionService.selectIds([...this.findAllChildren()]);
    };
    const selectNone = () => {
      this.selectionService.selectIds([]);
    };
    const noAction = () => {};

    const hasParents = this.findAllParents().length > 1;
    const hasChildren = this.findAllChildren().length > 1;
    const hasRelatives = hasParents || hasChildren;
    const pointSelected = this.selectionService.selectedIds.length > 0;

    return [
      {
        itemText: 'Random',
        displayIcon: false,
        menu: [],
        onClick: selectRandomDatapoint,
        disabled: false,
      },
      {
        itemText: 'All related',
        displayIcon: false,
        menu: [],
        onClick: hasRelatives ? selectAllRelatedCallback :
                                noAction,  // make this check more specific
        disabled: !hasRelatives,
      },
      {
        itemText: 'Parents',
        displayIcon: false,
        menu: [],
        onClick: hasParents ? selectAllParentsCallback : noAction,
        disabled: !hasParents,
      },
      {
        itemText: 'Children',
        displayIcon: false,
        menu: [],
        onClick: hasChildren ? selectAllChildrenCallback : noAction,
        disabled: !hasChildren,
      },
      this.getSliceItem(),
      // TODO(lit-dev): Add find siblings option.
      {
        itemText: 'Clear selection',
        displayIcon: false,
        menu: [],
        onClick: selectNone,
        disabled: !pointSelected,
      },
    ];
  }

  // Returns items for the datapoint menu.
  getColorOptionItems() {
    const colorOptions = this.colorService.colorableOptions.map((option) => {
      const isSelected =
          option.name === this.colorService.selectedColorOption.name;
      return {
        itemText: option.name,
        displayIcon: isSelected,
        menu: [],
        onClick: () => {
          this.colorService.selectedColorOptionName = option.name;
        },
        disabled: false,
      };
    });

    return colorOptions;
  }

  // Returns items for the slice menu.
  getSliceItem() {
    const sliceNames = Array.from(this.sliceService.namedSlices.keys());
    const sliceItems = sliceNames.map((name) => {
      const isSelected = this.sliceService.selectedSliceName === name;
      return {
        itemText: name,
        displayIcon: isSelected,
        menu: [],
        onClick: () => {
          this.sliceService.selectNamedSlice(name);
        },
        disabled: false
      };
    });

    return {
      itemText: 'Slices',
      displayIcon: false,
      menu: sliceItems,
      onClick: () => {},
      disabled: false,
    };
  }

  // Returns menu data for the menu toolbar.
  getMenuData() {
    const menuData = new Map<string, MenuItem[]>();
    menuData.set('Select datapoint', this.getSelectItems());
    menuData.set('Color by', this.getColorOptionItems());
    return menuData;
  }

  override render() {
    return html`<lit-menu-toolbar .menuData=${
        this.getMenuData()}></lit-menu-toolbar>`;
  }
}

/**
 * The selection controls toolbar
 */
@customElement('lit-main-toolbar')
export class LitMainToolbar extends MobxLitElement {
  static override get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');
  private readonly selectionService = app.getService(SelectionService);

  /**
   * ID pairs, as [child, parent], from current selection or the whole dataset.
   */
  @computed
  get selectedIdPairs() {
    const data = this.selectionService.selectedOrAllInputData;
    return data.filter(d => d.meta['parentId'])
        .map(d => [d.id, d.meta['parentId']!]);
  }

  /**
   * Button to select a random example.
   */
  renderLuckyButton() {
    const selectRandom = () => {
      const nextIndex = randInt(0, this.appState.currentInputData.length);
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    // clang-format off
    return html`
      <button id='select-random-button' class='hairline-button xl'
        @click=${selectRandom}>
        <span id='select-random-text'>Select random</span>
      </button>
    `;
    // clang-format on
  }

  /**
   * Controls to navigate through the dataset.
   */
  renderDatapointNavigation() {
    const numTotalDatapoints = this.appState.currentInputData.length;

    const primaryId = this.selectionService.primarySelectedId;
    let selectedIndex =
        this.appState.currentInputData.findIndex(ex => ex.id === primaryId);

    // Controls navigate through all datapoints by default.
    let propertyText = 'datapoints';
    let selectOffset = (offset: number) => {
      const nextIndex = primaryId == null ?
          0 :
          (selectedIndex + offset + numTotalDatapoints) % numTotalDatapoints;
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };

    const numSelected = this.selectionService.selectedIds.length;
    // If multiple items are selected, navigate through the selection instead.
    if (numSelected > 1 && primaryId !== null) {
      selectedIndex =
          this.selectionService.selectedIds.findIndex(id => id === primaryId);
      propertyText = 'selected';
      selectOffset = (offset: number) => {
        const nextIndex = (selectedIndex + offset + numSelected) % numSelected;
        const nextId = this.selectionService.selectedIds[nextIndex];
        this.selectionService.setPrimarySelection(nextId);
      };
    }

    // Set the selected set to the previous (-1) or next (1) example in
    // the dataset.
    const iconClass = classMap({'icon-button': true});
    // clang-format off
    return html`
      <mwc-icon class=${iconClass} id='ds-select-prev'
        @click=${() => {selectOffset(-1);}}>
        chevron_left
      </mwc-icon>
      <div id='number-selected'>
        <span id='num-selected-text'>${numSelected}</span> of <span>${numTotalDatapoints}</span> ${propertyText}
      </div>
      <mwc-icon class=${iconClass} id='ds-select-next'
        @click=${() => {selectOffset(1);}}>
        chevron_right
      </mwc-icon>
    `;
    // clang-format on
  }

  /**
   * Render a Slices button to show the Slice Editor.
   */
  renderSlices() {
    // Left-anchor the Slice Editor popup.
    const popupStyle = styleMap({'--popup-top': '4px'});
    // clang-format off
    return html`
    <popup-container style=${popupStyle}>
      <button class='hairline-button xl' slot='toggle-anchor-closed'>
        <span class='material-icon-outlined'>dataset</span>
        &nbsp;Slices&nbsp;
        <span class='material-icon'>expand_more</span>
      </button>
      <button class='hairline-button xl' slot='toggle-anchor-open'>
          <span class='material-icon-outlined'>dataset</span>
          &nbsp;Slices&nbsp;
          <span class='material-icon'>expand_less</span>
        </button>
      <div class='slice-container'>
        <lit-slice-module model="unused" shouldReact=1 selectionServiceIndex=0>
        </lit-slice-module>
      </div>
    </popup-container>
    `;
    // clang-format on
  }

  /**
   * Pair controls. Assumes reference is showing parent, main showing child.
   */
  renderPairControls() {
    const idPairs = this.selectedIdPairs;
    if (idPairs.length === 0) return null;

    const referenceSelectionService =
        app.getService(SelectionService, 'pinned');
    const maybeGetIndex = (id: string|null) =>
        id != null ? this.appState.indicesById.get(id) : null;
    // ID pairs, as [child, parent]
    const primaryIds = [
      this.selectionService.primarySelectedId,
      referenceSelectionService.primarySelectedId,
    ];
    // Index pairs, as [child, parent]
    const primaryIndices = primaryIds.map(maybeGetIndex);

    // Index into idPairs, if found.
    const selectedPairIndex = primaryIds.includes(null) ?
        -1 :
        idPairs.findIndex(
            pair => (compareArrays(pair, primaryIds as string[]) === 0));
    // True if the current selection is a counterfactual pair, with the
    // reference showing the parent.
    const isPairSelected = selectedPairIndex !== -1;
    const numPairs = idPairs.length;

    // Cycle through pairs, relative to the current selected one.
    const selectOffset = (offset: number) => {
      // If not in comparison mode, enter it.
      this.appState.compareExamplesEnabled = true;

      // If no valid pair is selected (-1), ignore and start at 0.
      const start = isPairSelected ? selectedPairIndex : 0;
      const nextPairIndex = (start + offset + numPairs) % numPairs;
      // [child, parent]
      const nextIds = idPairs[nextPairIndex];
      // Special case: if no points were selected, select all the pairs
      // so that we can continue to cycle through this set.
      if (this.selectionService.selectedIds.length === 0) {
        this.selectionService.selectIds(flatten(idPairs));
      }
      // Select child in main selection, parent in reference.
      this.selectionService.setPrimarySelection(nextIds[0]);
      referenceSelectionService.setPrimarySelection(nextIds[1]);
    };

    // Note: we're using the reference selection for the parent,
    // so show that as the first element in the pair to be consistent with the
    // rest of the UI.
    // clang-format off
    return html`
      <div id='pair-controls' class='selection-status-group'>
        ${numPairs} ${numPairs === 1 ? "pair" : "pairs"} available
        <mwc-icon class='icon-button' id='select-prev'
          @click=${() => {selectOffset(-1);}}>
          chevron_left
        </mwc-icon>
        <mwc-icon class='icon-button mdi-outlined' id='select-random'
          @click=${() => {selectOffset(randInt(1, numPairs));}}>
          casino
        </mwc-icon>
        <mwc-icon class='icon-button' id='select-next'
          @click=${() => {selectOffset(1);}}>
          chevron_right
        </mwc-icon>
        ${isPairSelected ?
        html`[${primaryIndices[1]}, ${primaryIndices[0]}] selected` : null}
      </div>
    `;
    // clang-format on
  }

  override render() {
    const clearSelection = () => {
      this.selectionService.selectIds([]);
    };
    const selectAll = () => {
      const allIds = this.appState.currentInputData.map(example => example.id);
      this.selectionService.selectIds(allIds);
    };
    const numSelected = this.selectionService.selectedIds.length;
    const primaryId = this.selectionService.primarySelectedId;
    const primaryIndex =
        primaryId == null ? -1 : this.appState.indicesById.get(primaryId)!;

    const updatePinnedDatapoint = () => {
      if (this.pinnedSelectionService.primarySelectedId) {
        this.pinnedSelectionService.selectIds([]);
        this.appState.compareExamplesEnabled = false;
      } else if (primaryId) {
        this.pinnedSelectionService.selectIds([primaryId]);
        this.appState.compareExamplesEnabled = true;
      }
    };

    let pinnedIndex = -1;
    if (this.appState.compareExamplesEnabled &&
        this.pinnedSelectionService.primarySelectedId) {
      pinnedIndex = this.appState.indicesById.get(
          this.pinnedSelectionService.primarySelectedId)!;
    }

    const title = this.appState.compareExamplesEnabled ?
        `Unpin datapoint ${pinnedIndex}` :
        primaryId == null ? 'Pin selected datapoint' :
                            `Pin datapoint ${primaryIndex}`;
    const pinDisabled =
        !this.appState.compareExamplesEnabled && primaryId == null;
    const pinClasses = classMap({
      'material-icon': true,
      'span-outlined': !this.appState.compareExamplesEnabled
    });
    const buttonClasses = classMap({
      'hairline-button': true,
      'xl': true,
      'pin-button': true,
      'active': this.appState.compareExamplesEnabled
    });
    // clang-format off
    return html`
    <div class='toolbar main-toolbar'>
      <div id='left-container'>
        <lit-main-menu></lit-main-menu>
        ${this.renderSlices()}
        <button class="${buttonClasses}" title=${title}
                ?disabled=${pinDisabled} @click=${updatePinnedDatapoint}>
          <div class="pin-button-content">
            <span class="${pinClasses}">push_pin</span>
            <div class="pin-button-text">${title}</div>
          </div>
        </button>
        ${this.renderPairControls()}
      </div>
      <div id='right-container'>
        ${this.renderDatapointNavigation()}
        <button id="select-all" class="hairline-button xl"
          @click=${selectAll}>
          Select all
        </button>
        ${this.renderLuckyButton()}
        <button id="clear-selection" class="hairline-button xl"
          @click=${clearSelection}
          ?disabled="${numSelected === 0}">
          Clear selection
        </button>
      </div>
    </div>
    `;
    // clang-format on
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'lit-main-toolbar': LitMainToolbar;
  }
}
