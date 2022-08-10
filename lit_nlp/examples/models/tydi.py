"""LIT wrappers for TyDiModel"""
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from typing import Optional, Dict, List
import numpy as np
import transformers



BertTokenizer = transformers.BertTokenizer
FlaxBertForQuestionAnswering = transformers.FlaxBertForQuestionAnswering
JsonDict = lit_types.JsonDict


class TyDiModel(lit_model.Model):
  """Question Answering Jax model based on TyDiQA Dataset ."""

  TYDI_LANG_VOCAB = ['english','bengali', 'russian', 'telugu','swahili',
                    'korean','indonesian','arabic','finnish']
                  
  def __init__(self,
              model_name: str,
              model=None,
              tokenizer=None,
              **config_kw):
    super().__init__()
    self.tokenizer = tokenizer or BertTokenizer.from_pretrained(model_name)
    self.model = model or FlaxBertForQuestionAnswering.from_pretrained(
        model_name)

  def _segment_slicers(self, tokens: List[str]):
    """Slicers along the tokens dimension for each segment.

    For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'],
    we want to get the slices [a0, a1, ...] and [b0, b1, ...]

    Args:
      tokens: <string>[num_tokens], including special tokens

    Returns:
      (slicer_a, slicer_b), slice objects
    """
    try:
      split_point = tokens.index(self.tokenizer.sep_token)
    except ValueError:
      split_point = len(tokens) - 1
    slicer_question = slice(1, split_point)  # start after [CLS]
    slicer_context = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
    return slicer_question, slicer_context

  def max_minibatch_size(self) -> int:
    return 8


  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples.

    Takes question & context from the dataset tokenizes &
    predicts answer.
    
    """
    prediction_output = []

    for inp in inputs:
      tokenized_text = self.tokenizer(
          inp['question'], inp['context'], return_tensors="jax", padding=True)
      results = self.model(
          **tokenized_text, output_hidden_states=True)
      answer_start_index = results.start_logits.argmax()
      answer_end_index = results.end_logits.argmax()
      predict_answer_tokens = tokenized_text.input_ids[
          0, answer_start_index : answer_end_index + 1]

      # get id's for question & context
      tokens = np.asarray(tokenized_text['input_ids'])
      #convert id's to tokens
      total_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
      #split by question & context
      slicer_question, slicer_context = self._segment_slicers(total_tokens)
      #get embeddings
      all_hidden_state = results.hidden_states[-1][0]
      
      prediction_output.append({
          "generated_text" : self.tokenizer.decode(predict_answer_tokens),
          "answers_text": inp['answers_text'],
          "cls_emb": results.hidden_states[-1][:, 0][0],  # last layer, first token,
          "tokens_question": total_tokens[slicer_question],
          "token_grad_question": all_hidden_state[slicer_question],
          "tokens_answer": total_tokens[slicer_context],
          "token_grad_answer": all_hidden_state[slicer_context]
      })
  
    return prediction_output


  def input_spec(self):
    return {
        "title":lit_types.TextSegment(),
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
        "answers_text": lit_types.MultiSegmentAnnotations(),
        "language": lit_types.CategoryLabel(required=False,vocab=self.TYDI_LANG_VOCAB),

    }

  def output_spec(self):
    return {
        "generated_text": lit_types.GeneratedText(parent='answers_text'),
         "cls_emb": lit_types.Embeddings(),
         "tokens_question": lit_types.Tokens(parent='question'),
         "tokens_answer": lit_types.Tokens(parent='question'),
         "token_grad_question" : lit_types.TokenGradients(
          align="tokens_question"),
          "token_grad_answer" : lit_types.TokenGradients(
          align="tokens_answer")

    }
