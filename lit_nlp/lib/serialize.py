# Copyright 2020 Google LLC
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
# Lint as: python3
"""Miscellaneous utility functions."""
import json
from typing import cast, Optional, Text

from lit_nlp.api import dtypes
from lit_nlp.api import types
import numpy as np

JsonDict = types.JsonDict


def _obj_to_json(o: object):
  """JSON serialization helper."""
  if isinstance(o, np.ndarray):
    return {
        '__class__': 'np.ndarray',
        '__value__': o.tolist(),
    }
  elif isinstance(o, np.number):
    # Handle numpy scalar types, like np.float32
    # This discards some precision information, but is consistent with using
    # .tolist() on a NumPy array.
    return cast(np.number, o).tolist()  # to regular Python scalar
  elif isinstance(o, types.LitType):
    return o.to_json()
  elif isinstance(o, dtypes.DataTuple):
    return o.to_json()
  elif isinstance(o, tuple):
    return {
        '__class__': 'tuple',
        '__value__': list(o),
    }
  else:
    raise TypeError(repr(o) + ' is not JSON serializable.')


# TODO(lit-team): remove this once frontend can use the invertible versions.
def _obj_to_json_simple(o: object):
  """JSON serialization helper. Not invertible!"""
  if isinstance(o, np.ndarray):
    return o.tolist()
  elif isinstance(o, np.number):
    # Handle numpy scalar types, like np.float32
    # This discards some precision information, but is consistent with using
    # .tolist() on a NumPy array.
    return cast(np.number, o).tolist()  # to regular Python scalar
  elif isinstance(o, types.LitType):
    return o.to_json()
  elif isinstance(o, dtypes.DataTuple):
    return o.to_json()
  elif isinstance(o, tuple):
    return list(o)
  else:
    raise TypeError(repr(o) + ' is not JSON serializable.')


def _obj_from_json(d: JsonDict):
  """JSON deserialization helper."""
  obj_class = d.pop('__class__', None)
  if obj_class == 'np.ndarray':
    return np.array(d['__value__'])
  elif obj_class == 'LitType':
    return types.LitType.from_json(d)
  elif obj_class == 'DataTuple':
    return dtypes.DataTuple.from_json(d)
  elif obj_class == 'tuple':
    return tuple(d['__value__'])
  else:
    return d


##
# Custom encoder classes for using built-in Python3 json library.
# This is a bit clunkier than simplejson, but has the big advantage of
# preserving key order in Python 3.
class SimpleJSONEncoder(json.JSONEncoder):

  def default(self, obj):
    return _obj_to_json_simple(obj)


class CustomJSONEncoder(json.JSONEncoder):

  def default(self, obj):
    return _obj_to_json(obj)


def from_json(json_string: Text) -> Optional[JsonDict]:
  """Reconstruct from a JSON string."""
  if json_string:
    return json.loads(json_string, object_hook=_obj_from_json)
  return None


def to_json(obj, simple=False, **json_kw) -> Text:
  """Serialize to a JSON string."""
  return json.dumps(
      obj, cls=SimpleJSONEncoder if simple else CustomJSONEncoder, **json_kw)
