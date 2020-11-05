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

import '@material/mwc-icon';
import './slice_toolbar';

// tslint:disable:no-new-decorators
import {MobxLitElement} from '@adobe/lit-mobx';
import {customElement, html} from 'lit-element';
import {classMap} from 'lit-html/directives/class-map';
import {styleMap} from 'lit-html/directives/style-map';
import {computed, observable} from 'mobx';

import {app} from '../core/lit_app';
import {MenuItem} from '../elements/menu';
import {compareArrays, flatten, randInt, shortenId} from '../lib/utils';
import {AppState, ColorService, SelectionService, SliceService} from '../services/services';
import {FAVORITES_SLICE_NAME} from '../services/slice_service';

import {styles} from './main_toolbar.css';
import {styles as sharedStyles} from './shared_styles.css';


/**
 * The selection controls toolbar
 */
@customElement('lit-main-menu')
export class LitMainMenu extends MobxLitElement {
  static get styles() {
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
    const selectAllFavoritedCallback = () => {
      this.sliceService.selectNamedSlice(FAVORITES_SLICE_NAME);
    };
    const selectNone = () => {
      this.selectionService.selectIds([]);
    };
    const noAction = () => {};

    const hasStarredPoints =
        !this.sliceService.isSliceEmpty(FAVORITES_SLICE_NAME);
    const hasParents = this.findAllParents().length > 1;
    const hasChildren = this.findAllChildren().length > 1;
    const hasRelatives = hasParents || hasChildren;
    const pointSelected = this.selectionService.selectedIds.length > 0;

    return [
      {
        itemText: 'Random datapoint',
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
        itemText: 'Parent datapoints',
        displayIcon: false,
        menu: [],
        onClick: hasParents ? selectAllParentsCallback : noAction,
        disabled: !hasParents,
      },
      {
        itemText: 'Children datapoints',
        displayIcon: false,
        menu: [],
        onClick: hasChildren ? selectAllChildrenCallback : noAction,
        disabled: !hasChildren,
      },
      {
        itemText: 'Favorite datapoints',
        displayIcon: false,
        menu: [],
        onClick: hasStarredPoints ? selectAllFavoritedCallback : noAction,
        disabled: !hasStarredPoints
      },
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
          this.colorService.selectedColorOption = option;
        },
        disabled: false,
      };
    });

    return colorOptions;
  }

  // Returns items for the slice menu.
  getSliceItems() {
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

    return [
      {
        itemText: 'Select slice',
        displayIcon: false,
        menu: sliceItems,
        onClick: () => {},
        disabled: false,
      },
    ];
  }

  // Returns menu data for the menu toolbar.
  getMenuData() {
    const menuData = new Map<string, MenuItem[]>();
    menuData.set('Select', this.getSelectItems());
    menuData.set('Color by', this.getColorOptionItems());
    menuData.set('Slices', this.getSliceItems());
    return menuData;
  }

  render() {
    return html`<lit-menu-toolbar .menuData=${
        this.getMenuData()}></lit-menu-toolbar>`;
  }
}

/**
 * The selection controls toolbar
 */
@customElement('lit-main-toolbar')
export class LitMainToolbar extends MobxLitElement {
  static get styles() {
    return [sharedStyles, styles];
  }

  private readonly appState = app.getService(AppState);
  private readonly selectionService = app.getService(SelectionService);
  private readonly sliceService = app.getService(SliceService);

  @observable private sliceToolbarVisible: boolean = false;
  @observable private highlightFavorite: boolean = false;

  /**
   * ID pairs, as [child, parent], from current selection or the whole dataset.
   */
  @computed
  get selectedIdPairs() {
    const data = this.selectionService.selectedOrAllInputData;
    return data.filter(d => d.meta['parentId'])
        .map(d => [d.id, d.meta['parentId']]);
  }

  private toggleFavorited() {
    const data = this.selectionService.primarySelectedInputData;
    if (data == null) {
      return;
    }

    if (data.meta['isFavorited'] != null) {
      data.meta['isFavorited'] = !data.meta['isFavorited'];

      if (data.meta['isFavorited']) {
        this.sliceService.addIdsToSlice(FAVORITES_SLICE_NAME, [data.id]);
      } else {
        this.sliceService.removeIdsFromSlice(FAVORITES_SLICE_NAME, [data.id]);
      }

      // If the favorites slice is selected, update the selection to include
      // the addition/removal of this id.
      if (this.sliceService.selectedSliceName === FAVORITES_SLICE_NAME) {
        this.sliceService.selectNamedSlice(FAVORITES_SLICE_NAME);
      }

      this.highlightFavorite = data.meta['isFavorited'];
    }
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
      <button id='select-random-button' class='text-button'
        @click=${selectRandom}>
        <mwc-icon class='mdi-outlined icon-button' id='select-random-icon'>
          casino
        </mwc-icon>
        <span id='select-random-text'>Select Random</span>
      </button>
    `;
    // clang-format on
  }


  /**
   * Controls to page through the dataset.
   * Assume exactly one point is selected.
   */
  renderSelectionArrows() {
    const primaryId = this.selectionService.primarySelectedId;
    const selectedIndex =
        this.appState.currentInputData.findIndex(ex => ex.id === primaryId);
    const numExamples = this.appState.currentInputData.length;

    // Set the selected set to the previous (-1) or next (1) example in the
    // dataset.
    const selectOffset = (offset: number) => {
      const nextIndex = (selectedIndex + offset + numExamples) % numExamples;
      const nextId = this.appState.currentInputData[nextIndex].id;
      this.selectionService.selectIds([nextId]);
    };
    // clang-format off
    return html`
      <mwc-icon class='icon-button' id='ds-select-prev'
        @click=${() => {selectOffset(-1);}}>
        chevron_left
      </mwc-icon>
      <mwc-icon class='icon-button mdi-outlined' id='ds-select-random'
        @click=${() => {selectOffset(randInt(1, numExamples));}}>
        casino
      </mwc-icon>
      <mwc-icon class='icon-button' id='ds-select-next'
        @click=${() => {selectOffset(1);}}>
        chevron_right
      </mwc-icon>
    `;
    // clang-format on
  }

  /**
   * Primary selection display, including controls
   * to page the primary selection through the selected set.
   */
  renderPrimarySelectControls() {
    const numSelected = this.selectionService.selectedIds.length;
    const primaryId = this.selectionService.primarySelectedId;
    if (primaryId == null) return;

    const displayedPrimaryId = shortenId(primaryId);
    const primaryIndex = this.appState.indicesById.get(primaryId);

    const selectedIndex =
        this.selectionService.selectedIds.findIndex(id => id === primaryId);
    // Set the primary selection to the previous (-1) or next (1) example in the
    // current selected list.
    const selectOffset = (offset: number) => {
      const nextIndex = (selectedIndex + offset + numSelected) % numSelected;
      const nextId = this.selectionService.selectedIds[nextIndex];
      this.selectionService.setPrimarySelection(nextId);
    };
    // clang-format off
    return html`
      <div id='primary-selection-status' class='selection-status-group'>
        <span id='primary-text'>
          (primary:&nbsp;<span class='monospace'> ${displayedPrimaryId}</span>
          ... [${primaryIndex}]
        </span>
        ${numSelected > 1 ? html`
          <mwc-icon class='icon-button' id='select-prev'
            @click=${() => {selectOffset(-1);}}>
            chevron_left
          </mwc-icon>
          <mwc-icon class='icon-button mdi-outlined' id='select-random'
            @click=${() => {selectOffset(randInt(1, numSelected));}}>
            casino
          </mwc-icon>
          <mwc-icon class='icon-button' id='select-next'
            @click=${() => {selectOffset(1);}}>
            chevron_right
          </mwc-icon>
        ` : null}
         ${this.renderFavoriteButton()} )
      </div>
    `;
    // clang-format on
  }

  /**
   * Pair controls. Assumes reference is showing parent, main showing child.
   */
  renderPairControls() {
    const idPairs = this.selectedIdPairs;
    if (idPairs.length === 0) return null;

    const referenceSelectionService = app.getServiceArray(SelectionService)[1];
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
        ${numPairs} pairs available
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

  renderFavoriteButton() {
    const primarySelectedInputData =
        this.selectionService.primarySelectedInputData;
    if (primarySelectedInputData == null) {
      return;
    }
    this.highlightFavorite = primarySelectedInputData.meta['isFavorited'];

    const favoriteOnClick = () => {
      this.toggleFavorited();
    };
    return html`<mwc-icon class='icon-button' id='favorite-button' @click=${
        favoriteOnClick}>
            ${this.highlightFavorite ? 'favorite' : 'favorite_border'}
          </mwc-icon>`;
  }

  // Render the slice toolbar.
  renderSliceToolbar() {
    const toggleSliceToolbar = () => {
      this.sliceToolbarVisible = !this.sliceToolbarVisible;
    };
    const sliceToolbarStyle = styleMap(
        {'visibility': (this.sliceToolbarVisible ? 'visible' : 'hidden')});

    // clang-format off
    return html`
    <div class="toolbar-holder">
      <button id='toggle-slice-toolbar' @click=${toggleSliceToolbar}>
        Slices <span data-icon=${
          this.sliceToolbarVisible ? "expand_less" : "expand_more"}></span>
      </button>
      <div class='togglable-toolbar-holder' id='slice-toolbar-container'
        style=${sliceToolbarStyle}>
        <lit-slice-toolbar></lit-slice-toolbar>
      </div>
    </div>
    `;
    // clang-format on
  }

  render() {
    const clearSelection = () => {
      this.selectionService.selectIds([]);
    };
    const numSelected = this.selectionService.selectedIds.length;
    const numTotal = this.appState.currentInputData.length;
    const primaryId = this.selectionService.primarySelectedId;

    const compareExamplesClass =
        classMap({'mode-active': this.appState.compareExamplesEnabled});

    const toggleExampleComparison = () => {
      this.appState.compareExamplesEnabled =
          !this.appState.compareExamplesEnabled;
    };
    // clang-format off
    return html`
    <div class='toolbar' id='main-toolbar'>
      <lit-main-menu></lit-main-menu>
        <div id='status-group-left'>
        <div>
          ${numSelected} of ${numTotal} selected
        </div>
        ${numSelected === 0 ? this.renderLuckyButton() : null}
        ${numSelected === 1 ? this.renderSelectionArrows() : null}
        ${primaryId !== null ? this.renderPrimarySelectControls() :  null}
        ${this.appState.compareExamplesEnabled ? this.renderPairControls() : null}
        <button id="clear-selection" class="text-button" @click=${clearSelection}
          ?disabled="${numSelected === 0}">
          Clear selection
        </button>
        <button id='toggle-example-comparison'
          @click=${toggleExampleComparison} class=${compareExamplesClass}
          ?disabled="${!this.appState.compareExamplesEnabled && numSelected === 0}">
          Compare Datapoints
        </button>
        ${this.renderSliceToolbar()}
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
