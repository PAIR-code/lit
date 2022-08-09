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

  def _question_gradient_generator(self, question: List[str]):
    tokenized_text = self.tokenizer(question, return_tensors="jax", padding=True)
    question_output = self.model(
          **tokenized_text, output_attentions=True, output_hidden_states=True)
    tokens = np.asarray(tokenized_text['input_ids'])
    output = {}
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(tokens[0])
    return output["tokens"], question_output.hidden_states[-1][0]
  
  def _answer_gradient_generator(self, context: List[str]):
    tokenized_text = self.tokenizer(context, return_tensors="jax", padding=True)
    answer_output = self.model(
          **tokenized_text, output_attentions=True, output_hidden_states=True)
    tokens = np.asarray(tokenized_text['input_ids'])
    output = {}
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(tokens[0])
    return output["tokens"], answer_output.hidden_states[-1][0]
  
  def _segment_slicers(self, tokens: List[str]):
    try:
      split_point = tokens.index(self.tokenizer.sep_token)
    except ValueError:
      split_point = len(tokens) - 1
    slicer_question = slice(1, split_point)  # start after [CLS]
    slicer_answer = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
    return slicer_question, slicer_answer

  def max_minibatch_size(self) -> int:
    return 8


  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    prediction_output = []

    for inp in inputs:
      tokenized_text = self.tokenizer(
          inp['question'], inp['context'], return_tensors="jax", padding=True)
      results = self.model(
          **tokenized_text, output_attentions=True, output_hidden_states=True)
      answer_start_index = results.start_logits.argmax()
      answer_end_index = results.end_logits.argmax()
      predict_answer_tokens = tokenized_text.input_ids[
          0, answer_start_index : answer_end_index + 1]

      # tokenized_text_question =self.tokenizer(inp['question'], return_tensors="jax", padding=True)
      # question_output = self.model(
      #     **tokenized_text_question, output_attentions=True, output_hidden_states=True)
      # tokens = np.asarray(tokenized_text_question['input_ids'])
      # output ={}
      # output["tokens"] = self.tokenizer.convert_ids_to_tokens(tokens[0])

      # # Tokens for each segment, individually.
      # slicer_a, slicer_b = self._segment_slicers(output["tokens"])
      # output["tokens_question"] = output["tokens"][slicer_a]
      # output["input_emb_grad"] = question_output.hidden_states[-1][0]
      result_tokens = np.asarray(tokenized_text['input_ids'])
      output = {}
      output['tokens'] = self.tokenizer.convert_ids_to_tokens(result_tokens[0])
      slicer_question, slicer_answer = self._segment_slicers(output["tokens"])

      q_tokens, q_hidden_states = self._question_gradient_generator(inp['question'])
      a_tokens, a_hidden_states = self._question_gradient_generator(inp['context'])
      # print(q_tokens[slicer_question])
      prediction_output.append({
          "generated_text" : self.tokenizer.decode(predict_answer_tokens),
          "answers_text": inp['answers_text'],
          "cls_emb": results.hidden_states[-1][:, 0][0],  # last layer, first token,
          # "tokens_question" : output["tokens"][slicer_a],
          # "token_grad_question" : output["input_emb_grad"][slicer_a],
          "tokens_question": q_tokens[slicer_question],
          "token_grad_question": q_hidden_states[slicer_question],
          "tokens_answer": a_tokens[slicer_answer],
          "token_grad_answer": a_hidden_states[slicer_answer],
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
