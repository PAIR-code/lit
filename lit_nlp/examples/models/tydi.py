"""LIT wrappers for TyDiModel."""
from collections.abc import Iterable
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.datasets import question_answering
import numpy as np
import transformers


_BertTokenizer = transformers.BertTokenizer
_FlaxBertForQuestionAnswering = transformers.FlaxBertForQuestionAnswering
_JsonDict = lit_types.JsonDict


class TyDiModel(lit_model.Model):
  """Question Answering Jax model based on TyDiQA Dataset."""

  def __init__(
      self,
      model_name: str,
      model=None,
      tokenizer=None,
      **unused_kw,
  ):
    super().__init__()
    self.tokenizer = tokenizer or _BertTokenizer.from_pretrained(model_name)
    self.model = model or _FlaxBertForQuestionAnswering.from_pretrained(
        model_name
    )

  def _segment_slicers(self, tokens: list[str]):
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
    # Question starts after the [CLS] token
    slicer_question = slice(1, split_point)
    # Context ends before the last [SEP] token
    slicer_context = slice(split_point + 1, len(tokens) - 1)
    return slicer_question, slicer_context

  def max_minibatch_size(self) -> int:
    return 8

  def predict(self, inputs: Iterable[_JsonDict], **kw) -> Iterable[_JsonDict]:
    """Predict the answer given the question and context."""
    prediction_output: list[_JsonDict] = []

    for inp in inputs:
      tokenized_text = self.tokenizer(
          inp["question"], inp["context"], return_tensors="jax", padding=True
      )
      results = self.model(
          **tokenized_text, output_hidden_states=True
      )
      answer_start_index = results.start_logits.argmax()
      answer_end_index = results.end_logits.argmax()
      predict_answer_tokens = tokenized_text.input_ids[
          0, answer_start_index : answer_end_index + 1
      ]

      # get id's for question & context
      tokens = np.asarray(tokenized_text["input_ids"])
      # convert id's to tokens
      total_tokens = self.tokenizer.convert_ids_to_tokens(tokens[0])
      # split by question & context
      slicer_question, slicer_context = self._segment_slicers(total_tokens)
      # get embeddings
      embeddings = results.hidden_states[0][0]
      # gradient
      gradient = results.hidden_states[-1][0]

      prediction_output.append({
          "generated_text": self.tokenizer.decode(predict_answer_tokens),
          "answers_text": inp["answers_text"],
          # Embeddings come from the first token of the last layer.
          "cls_emb": results.hidden_states[-1][:, 0][0],
          "tokens_question": total_tokens[slicer_question],
          "tokens_context": total_tokens[slicer_context],
          "grad_class": None,
          "tokens_embs_question": np.asarray(embeddings[slicer_question]),
          "token_grad_context": np.asarray(embeddings[slicer_context]),
          "tokens_grad_question": np.asarray(gradient[slicer_question]),
          "tokens_embs_context": np.asarray(gradient[slicer_context])
      })

    return prediction_output

  def input_spec(self):
    return {
        "title": lit_types.TextSegment(),
        "context": lit_types.TextSegment(),
        "question": lit_types.TextSegment(),
        "answers_text": lit_types.MultiSegmentAnnotations(),
        "language": lit_types.CategoryLabel(
            required=False, vocab=question_answering.TYDI_LANG_VOCAB
        ),
    }

  def output_spec(self):
    return {
        "generated_text": lit_types.GeneratedText(parent="answers_text"),
        "cls_emb": lit_types.Embeddings(),
        "tokens_question": lit_types.Tokens(parent="question"),
        "tokens_embs_question": lit_types.TokenEmbeddings(
            align="tokens_question"
        ),
        "tokens_grad_question": lit_types.TokenGradients(
            align="tokens_question", grad_for="tokens_embs_question"
        ),
        "tokens_context": lit_types.Tokens(parent="question"),
        "tokens_embs_context": lit_types.TokenEmbeddings(
            align="tokens_context"
        ),
        "token_grad_context": lit_types.TokenGradients(
            align="tokens_context", grad_for="tokens_embs_context"
        ),
    }
