"""Data loaders for Question answering model."""
import re

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes
from lit_nlp.api import types as lit_types
import tensorflow_datasets as tfds


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


class TyDiQA(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self, split: str, max_examples=-1):
    ds = tfds.load('tydi_qa', split=split)

    # populate this with data records
    self._examples = []
    for row in ds.take(max_examples):
      answers_text = row['answers']['text'].numpy()
      answers_start = [row['answers']['answer_start'].numpy()[0]]
      answers = []
      # gets language id example: finnish--9069599​462862564793-0
      language_id = row['id'].numpy().decode('utf-8')
      alpha_chars_filter = re.findall(r'[a-z]', language_id)
      language = ''.join(str(r) for r in alpha_chars_filter)

      for label, start in zip(answers_text, answers_start):
        span = dtypes.SpanLabel(start, start + len(label), align='context')
        answers.append(
            dtypes.AnnotationCluster(label=label.decode('utf-8'), spans=[span])
        )

      self._examples.append({
          'answers_text': answers,
          'title': row['title'].numpy().decode('utf-8'),
          'context': row['context'].numpy().decode('utf-8'),
          'question': row['question'].numpy().decode('utf-8'),
          'language': language,
      })

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
