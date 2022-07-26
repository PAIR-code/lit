"""LIT wrappers for TyDiModel"""
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
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
      prediction_output.append({
          "generated_text" : self.tokenizer.decode(predict_answer_tokens),
          "answers_text": inp['answers_text']
      })
    
    return prediction_output


  def input_spec(self):
    return {
        "title":lit_types.TextSegment(),
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
        "answers_text": lit_types.MultiSegmentAnnotations(),
        "language": lit_types.CategoryLabel(required=False,vocab=self.TYDI_LANG_VOCAB)
    }

  def output_spec(self):
    return {
        "generated_text": lit_types.GeneratedText(parent='answers_text'),
    }
