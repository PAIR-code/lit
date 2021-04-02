# Lint as: python3
"""Wrapper for Stanza model"""

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.api import dtypes

SpanLabel = dtypes.SpanLabel
EdgeLabel = dtypes.EdgeLabel


class StanzaTagger(lit_model.Model):
  def __init__(self, model, tasks):
    self.model = model
    self.sequence_tasks = tasks["sequence"]
    self.span_tasks = tasks["span"]
    self.edge_tasks = tasks["edge"]

    self._input_spec = {
      "sentence": lit_types.TextSegment(),
    }

    self._output_spec = {
      "tokens": lit_types.Tokens(),
    }

    # Output spec based on specified tasks
    for task in self.sequence_tasks:
      self._output_spec[task] = lit_types.SequenceTags(align="tokens")
    for task in self.span_tasks:
      self._output_spec[task] = lit_types.SpanLabels(align="tokens")
    for task in self.edge_tasks:
      self._output_spec[task] = lit_types.EdgeLabels(align="tokens")

  def _predict(self, ex):
    """
    Predicts all specified tasks for an individual example
    :param ex (dict):
        This should be a dict with a single entry with:
            key = "sentence"
            value (str) = a single string for prediction
    :return (list):
        This list contains dicts for each prediction tasks with:
            key = task name
            value (list) = predictions
    """
    doc = self.model(ex["sentence"])
    prediction = {}
    for sentence in doc.sentences:
      prediction["tokens"] = [word.text for word in sentence.words]

      # Process each sequence task
      for task in self.sequence_tasks:
        prediction[task] = [word.to_dict()[task] for word in sentence.words]

      # Process each span task
      for task in self.span_tasks:
        # Mention is currently the only span task
        if task == "mention":
          prediction[task] = []
          for entity in sentence.entities:
            # Stanza indexes start/end of entities on char. LIT needs them as token indexes
            start, end = entity_char_to_token(entity, sentence)
            span_label = SpanLabel(start=start, end=end, label=entity.type)
            prediction[task].append(span_label)

      # Process each edge task
      for task in self.edge_tasks:
        # Deps is currently the only edge task
        if task == "deps":
          prediction[task] = []
          for relation in sentence.dependencies:
            label = relation[1]
            span1 = relation[2].id
            span2 = relation[2].id if label == "root" else relation[0].id
            edge_label = EdgeLabel(
              (span1 - 1, span1), (span2 - 1, span2), label
            )
            prediction[task].append(edge_label)

    return prediction

  def predict_minibatch(self, inputs, config=None):
    return [self._predict(ex) for ex in inputs]

  def input_spec(self):
    return self._input_spec

  def output_spec(self):
    return self._output_spec


def entity_char_to_token(entity, sentence):
  """
  Takes Stanza entity and sentence objects and returns the start and end tokens for the entity
  :param entity: Stanza entity
  :param sentence: Stanza sentence
  :return (int, int): Returns the start and end locations indexed by tokens
  """
  start_token, end_token = None, None
  for i, v in enumerate(sentence.words):
    x = v.misc.split("|")
    if "start_char=" + str(entity.start_char) in x:
      start_token = i
    if "end_char=" + str(entity.end_char) in x:
      end_token = i + 1
  return start_token, end_token
