"""Data loaders for Question answering model."""
import json

from huggingface_hub import hf_hub_download
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import types as lit_types

TYDI_LANG_VOCAB = [
    'english',
    'bengali',
    'russian',
    'telugu',
    'swahili',
    'korean',
    'indonesian',
    'arabic',
    'finnish',
]

# TyDi QA gold passages, mirrored per-language on HuggingFace.
# See https://huggingface.co/datasets/khalidalt/tydiqa-goldp
_TYDI_REPO_ID = 'khalidalt/tydiqa-goldp'

_LANG_BY_CODE = {
    'ar': 'arabic',
    'bn': 'bengali',
    'en': 'english',
    'fi': 'finnish',
    'id': 'indonesian',
    'ja': 'japanese',
    'ko': 'korean',
    'ru': 'russian',
    'sw': 'swahili',
    'te': 'telugu',
    'th': 'thai',
}


def _load_jsonl(split_dir: str, language: str) -> list[dict]:
  path = hf_hub_download(
      repo_id=_TYDI_REPO_ID,
      repo_type='dataset',
      filename=f'{split_dir}/{language}-{split_dir}.jsonl',
  )
  with open(path, encoding='utf-8') as f:
    return [json.loads(line) for line in f]


def _char_offset(text: str, byte_offset: int) -> int:
  """Convert a UTF-8 byte offset in text to a character offset."""
  return len(text.encode('utf-8')[:byte_offset].decode('utf-8', 'ignore'))


class TyDiQA(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self, split: str, max_examples=-1):
    if split == 'train':
      split_dir = 'train'
    elif split == 'validation':
      split_dir = 'dev'
    elif split.startswith('validation-'):
      code = split[len('validation-') :]
      if code not in _LANG_BY_CODE:
        raise ValueError(f'Unknown language code in split: {split}')
      split_dir = 'dev'
    else:
      raise ValueError(f'Unsupported split: {split}')

    languages = (
        [_LANG_BY_CODE[split[len('validation-') :]]]
        if split.startswith('validation-')
        else sorted(_LANG_BY_CODE.values())
    )

    # populate this with data records
    self._examples = []
    for language in languages:
      for row in _load_jsonl(split_dir, language):
        context = row['passage_text']
        answers = []
        for answer in row['answers']:
          label = answer['text']
          start = _char_offset(context, answer['start_byte'])
          span = dtypes.SpanLabel(start, start + len(label), align='context')
          answers.append(
              dtypes.AnnotationCluster(label=label, spans=[span])
          )

        self._examples.append({
            'answers_text': answers,
            'title': row['document_title'],
            'context': context,
            'question': row['question_text'],
            'language': row['language'],
        })
        if 0 <= max_examples <= len(self._examples):
          break
      if 0 <= max_examples <= len(self._examples):
        break
    if max_examples >= 0:
      self._examples = self._examples[:max_examples]

  def spec(self) -> lit_types.Spec:
    return {
        'title': lit_types.TextSegment(),
        'context': lit_types.TextSegment(),
        'question': lit_types.TextSegment(),
        'answers_text': lit_types.MultiSegmentAnnotations(),
        'language': lit_types.CategoryLabel(
            required=False, vocab=TYDI_LANG_VOCAB
        )
    }
