# Copyright 2021 Google LLC
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
"""Wrapper for Stanza model."""

from lit_nlp.api import dtypes
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types

SpanLabel = dtypes.SpanLabel
EdgeLabel = dtypes.EdgeLabel


class StanzaTagger(lit_model.Model):
  """Stanza Model wrapper."""

  def __init__(self, model, tasks):
    """Initialize with Stanza model and a dictionary of tasks.

    Args:
      model: A Stanza model
      tasks: A dictionary of tasks, grouped by task type.
        Keys are the grouping, which should be one of:
          ('sequence', 'span', 'edge').
        Values are a list of stanza task names as strings.
    """
    self.model = model
    # Store lists of task name strings by grouping
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
    """Predicts all specified tasks for an individual example.

    Args:
      ex (dict): This should be a dict with a single entry.
        key = "sentence"
        value (str) = a single string for prediction
    Returns:
        A list containing dicts for each prediction tasks with:
          key = task name
          value (list) = predictions
    Raises:
      ValueError: Invalid task name.
    """
    doc = self.model(ex["sentence"])
    prediction = {task: [] for task in self._output_spec}
    for sentence in doc.sentences:
      # Get starting token of the offset to align task for multiple sentences
      start_token = len(prediction["tokens"])
      prediction["tokens"].extend([word.text for word in sentence.words])

      # Process each sequence task
      for task in self.sequence_tasks:
        prediction[task].extend(
            [word.to_dict()[task] for word in sentence.words])

      # Process each span task
      for task in self.span_tasks:
        # Mention is currently the only span task
        if task == "mention":
          for entity in sentence.entities:
            # Stanza indexes start/end of entities on char. LIT needs them as
            # token indexes
            start, end = entity_char_to_token(entity, sentence)
            span_label = SpanLabel(
                start=start + start_token,
                end=end + start_token,
                label=entity.type)
            prediction[task].append(span_label)
        else:
          raise ValueError(f"Invalid span task: '{task}'")

      # Process each edge task
      for task in self.edge_tasks:
        # Deps is currently the only edge task
        if task == "deps":
          for relation in sentence.dependencies:
            label = relation[1]
            span1 = relation[2].id + start_token
            span2_index = 2 if label == "root" else 0
            span2 = relation[span2_index].id + start_token
            # Relation lists have a root value at index 0, so subtract 1 to
            # align them to tokens
            edge_label = EdgeLabel((span1 - 1, span1), (span2 - 1, span2),
                                   label)
            prediction[task].append(edge_label)
        else:
          raise ValueError(f"Invalid edge task: '{task}'")

    return prediction

  def predict_minibatch(self, inputs, config=None):
    return [self._predict(ex) for ex in inputs]

  def input_spec(self):
    return self._input_spec

  def output_spec(self):
    return self._output_spec


def entity_char_to_token(entity, sentence):
  """Takes Stanza entity and sentence objects and returns the start and end tokens for the entity.

  The misc value in a stanza sentence object contains a string with additional
  information, separated by a pipe character. This string contains the
  start_char and end_char for each token, along with other information. This is
  extracted and used to match the start_char and end char values in a span
  object to return the start and end tokens for the entity.

  Example entity:
    {'text': 'Barack Obama',
    'type': 'PERSON',
    'start_char': 0,
    'end_char': 13}
  Example sentence:
    [
      {'id': 1,
      'text': 'Barack',
      ...,
      'misc': 'start_char=0|end_char=7'},
      {'id': 2,
      'text': 'Obama',
      ...,
      'misc': 'start_char=8|end_char=13'}
    ]

  Args:
    entity: Stanza Span object
    sentence: Stanza Sentence object
  Returns:
    Returns the token index of start and end locations for the entity
  """
  start_token, end_token = None, None
  for i, v in enumerate(sentence.words):
    x = v.misc.split("|")
    if "start_char=" + str(entity.start_char) in x:
      start_token = i
    if "end_char=" + str(entity.end_char) in x:
      end_token = i + 1
  return start_token, end_token
