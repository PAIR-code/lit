# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Selection state tracker.

This is a stateful component, intended for use in notebook/Colab contexts
to sync the UI selection state back to Python for further analysis.
"""
from typing import Optional

import attr
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types

IndexedInput = types.IndexedInput
JsonDict = types.JsonDict


@attr.s(auto_attribs=True, kw_only=True)
class UIState(object):
  """UI state."""
  dataset_name: Optional[str] = None
  dataset: Optional[lit_dataset.IndexedDataset] = None

  primary: Optional[IndexedInput] = None
  selection: list[IndexedInput] = attr.Factory(list)
  pinned: Optional[IndexedInput] = None


class UIStateTracker(object):
  """UI state tracker; mirrors state from frontend SelectionService.

  WARNING: this component is _stateful_, and in current form implements no
  locking or access control. We recommend using this only in a single-user,
  single-threaded context such as IPython or Colab notebooks.
  """

  def __init__(self):
    self._state = UIState()

  @property
  def state(self):
    return self._state

  def update_state(self,
                   indexed_inputs: list[types.IndexedInput],
                   dataset: lit_dataset.IndexedDataset,
                   dataset_name: str,
                   primary_id: Optional[str] = None,
                   pinned_id: Optional[str] = None):
    """Update state from the UI."""

    self._state.dataset_name = dataset_name
    self._state.dataset = dataset

    if primary_id:
      self._state.primary = dataset.index[primary_id]
    else:
      self._state.primary = None

    self._state.selection = indexed_inputs

    if pinned_id:
      self._state.pinned = dataset.index[pinned_id]
    else:
      self._state.pinned = None
