"""Data loaders for summarization datasets."""

from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.api import dtypes
import tensorflow_datasets as tfds


class TyDiQA(lit_dataset.Dataset):
  """TyDiQA dataset."""

  def __init__(self, split: str, max_examples=-1):
    """Dataset constructor, loads the data into memory."""
    ds = tfds.load("tydi_qa", split=split)
 

    # populate this with data records
    self._examples = []
    for row in ds.take(max_examples):
      answers_text = row['answers']['text'].numpy()
      answers_start = row['answers']['answer_start'].numpy()
      answers = []

      for label, start in zip(answers_text, answers_start):
        span = dtypes.SpanLabel(start, start + len(label))
        answers.append(dtypes.AnnotationCluster(label=label.decode(), spans=[span]))

      self._examples.append({
        'answers_text': answers,
        'title': row['title'].numpy().decode('utf-8'),
        'context': row['context'].numpy().decode('utf-8'),
        'question': row['question'].numpy().decode('utf-8'),
      })


        
      

  def spec(self) -> lit_types.Spec:
    """Dataset spec, which should match the model"s input_spec()."""
    return {
        "title":lit_types.TextSegment(),
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
        "answers_text": lit_types.MultiSegmentAnnotations() 

    }