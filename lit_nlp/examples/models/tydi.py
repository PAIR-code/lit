"""LIT wrappers for T5, supporting both HuggingFace and SavedModel formats."""
import re
from typing import List

import attr
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils

import tensorflow as tf
import numpy as np
# tensorflow_text is required for T5 SavedModel
import tensorflow_text  # pylint: disable=unused-import
import transformers

from rouge_score import rouge_scorer



class TyDiModel(lit_model.Model):
  """Question Answering Jax model based on TyDiQA Dataset ."""

  # Mapping from generic T5 fields to this task
  FIELD_RENAMES = {
      "input_text": "context",
      "target_text": "question",
  }
  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, 
              model_name="mrm8488/bert-multi-cased-finedtuned-xquad-tydiqa-goldp", 
              model=None,
              tokenizer=None,
              **config_kw):
    super().__init__()
    # self.config = TyDiModelConfig(**config_kw)
    self.tokenizer = tokenizer or BertTokenizer.from_pretrained(model_name)
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    
    self.model = model or FlaxBertForQuestionAnswering.from_pretrained(model_name)

    # # TODO(gehrmann): temp solution for ROUGE.
    self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # If output is List[(str, score)] instead of just str
    self._multi_output = isinstance(self.output_spec()["output_text"],
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

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # tokenize the text. -> then return prediction
  
    # Text as sequence of sentencepiece ID"s.
    context =[]
    question = []
    for i in inputs:
        question.append(i['question'])
        context.append(i['context'])

    prediction_output = []

    for i in range(len(inputs)):
    
        tokenized_text = self.tokenizer(question[i], context[i], return_tensors="jax",padding=True)
        results = self.model(**tokenized_text, output_attentions=True, output_hidden_states=True)

        answer_start_index = results.start_logits.argmax()

        answer_end_index = results.end_logits.argmax()

        predict_answer_tokens = tokenized_text.input_ids[0, answer_start_index : answer_end_index + 1]

        # creating output Dict
        output = {
            "output_text" : self.tokenizer.decode(predict_answer_tokens),
            "tokens": self.tokenizer.convert_ids_to_tokens(tokenized_text.input_ids[i]),

        }
        # Explanation for attentions
        for j in range(len(results.attentions)):
          output[f"layer_{j+1:d}_attention"] = str(results.attentions[j])
        
        for j in range(len(results.hidden_states)):
          output[f"layer_{j:d}_avg_embedding"] = tf.math.reduce_mean(
              results.hidden_states[i], axis=1)
        prediction_output.append(output)

   
    # Getting ROUGE scores
    for ex, mo in zip(inputs, prediction_output):
      score = self._scorer.score(
          target=ex["context"],
          prediction=self._get_pred_string(mo["output_text"]))
      mo["rougeL"] = float(score["rougeL"].fmeasure)
      yield mo


  def input_spec(self):
    return {
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    ret = {
        "output_text": lit_types.GeneratedText(parent="question"),
        "rougeL": lit_types.Scalar(),
        "tokens": lit_types.Tokens(parent="context"),
    }
    # Add attention and embeddings from each layer.
    for i in range(self.model.config.num_hidden_layers):
      ret[f"layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
          align_in="tokens", align_out="tokens")
      ret[f"layer_{i:d}_avg_embedding"] = lit_types.Embeddings()
    return ret
  
