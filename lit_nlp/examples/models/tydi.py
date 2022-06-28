"""LIT wrappers for TyDiModel"""
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils

import tensorflow as tf
import numpy as np

import transformers

from rouge_score import rouge_scorer

BertTokenizer = transformers.BertTokenizer
FlaxBertForQuestionAnswering = transformers.FlaxBertForQuestionAnswering
JsonDict = lit_types.JsonDict

class TyDiModel(lit_model.Model):
  """Question Answering Jax model based on TyDiQA Dataset ."""


  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, 
              model_name: str, 
              model=None,
              tokenizer=None,
              **config_kw):
    super().__init__()
    # self.config = TyDiModelConfig(**config_kw)
    self.tokenizer = tokenizer or BertTokenizer.from_pretrained(model_name)
    
    self.model = model or FlaxBertForQuestionAnswering.from_pretrained(model_name)

    # # TODO(gehrmann): temp solution for ROUGE.
    self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # If output is List[(str, score)] instead of just str
    self._multi_output = isinstance(self.output_spec()["generated_text"],
                                    lit_types.GeneratedTextCandidates)
    self._get_pred_string = (
        lit_types.GeneratedTextCandidates.top_text if self._multi_output else
        (lambda x: x))
  ##
  # LIT API implementation
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  # def findSpecKeys(spec, type):


  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # tokenize the text. -> then return prediction
  
    # Text as sequence of sentencepiece ID"s.
    context =[]
    question = []
    answers_text = []
    for i in inputs:
        question.append(i['question'])
        context.append(i['context'])
        answers_text.append(i['answers_text'])

    prediction_output = []

    for i in range(len(inputs)):
    
        tokenized_text = self.tokenizer(question[i], context[i], return_tensors="jax",padding=True)
        results = self.model(**tokenized_text, output_attentions=True, output_hidden_states=True)

        answer_start_index = results.start_logits.argmax()

        answer_end_index = results.end_logits.argmax()

        predict_answer_tokens = tokenized_text.input_ids[0, answer_start_index : answer_end_index + 1]

        # creating output Dict
        output = {
            "generated_text" : self.tokenizer.decode(predict_answer_tokens),
            "generated_text2" : self.tokenizer.decode(predict_answer_tokens),
            # adding answer_text for debugging
            "answers_text": answers_text[i]
            # "tokens": self.tokenizer.convert_ids_to_tokens(tokenized_text.input_ids[i]),

        }
        prediction_output.append(output)

    # printing for debugging
    # print('Answers------->\n')
    # print(prediction_output)
    return prediction_output
    # Getting ROUGE scores
    # for ex, mo in zip(inputs, prediction_output):
    #   score = self._scorer.score(
    #       target=ex["context"],
    #       prediction=self._get_pred_string(mo["generated_text"]))
    #   mo["rougeL"] = float(score["rougeL"].fmeasure)
    #   yield mo


  def input_spec(self):
    return {
        "title":lit_types.TextSegment(),
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
        "answers_text": lit_types.MultiSegmentAnnotations(),
    }

  def output_spec(self):

    return {
        # "generated_text": lit_types.GeneratedText(parent="context"),
        # "rougeL": lit_types.Scalar(),
        "generated_text": lit_types.GeneratedText(parent='answers_text'),
    }
    
  
