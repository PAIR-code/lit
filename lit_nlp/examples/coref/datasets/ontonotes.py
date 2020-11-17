"""Loader for OntoNotes coreference data."""
import json

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.api import types as lit_types

EdgeLabel = lit_dtypes.EdgeLabel


class OntonotesCorefDataset(lit_dataset.Dataset):
  """OntoNotes coreference data, from edge probing format.

  To get the edge probing data, see instructions at
  https://github.com/nyu-mll/jiant-v1-legacy/tree/master/probing/data#ontonotes
  """

  def convert_ep_record(self, record):
    """Convert edge probing record to LIT inputs."""
    edges = [
        EdgeLabel(span1=t['span1'], span2=t['span2'], label=int(t['label']))
        for t in record['targets']
    ]
    return {
        'text': record['text'],
        'tokens': record['text'].split(),
        'coref': edges,
    }

  def __init__(self, edgeprobe_json_path: str):
    with open(edgeprobe_json_path) as fd:
      ep_records = [json.loads(line) for line in fd]

    self._examples = [self.convert_ep_record(r) for r in ep_records]

  def spec(self):
    return {
        'text': lit_types.TextSegment(),
        'tokens': lit_types.Tokens(parent='text'),
        'coref': lit_types.EdgeLabels(align='tokens'),
    }
