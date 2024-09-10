"""Constants used across parallel classes in the Prompt Debugging example."""

import types
from lit_nlp.api import types as lit_types


class FieldNames(types.SimpleNamespace):
  PROMPT = "prompt"
  RESPONSE = "response"
  PROMPT_EMBEDDINGS = "prompt_embeddings"
  RESPONSE_EMBEDDINGS = "response_embeddings"
  TARGET = "target"
  TOKENS = "tokens"
  TARGET_MASK = "target_mask"
  GRAD_DOT_INPUT = "grad_dot_input"
  GRAD_NORM = "grad_l2"


INPUT_SPEC: lit_types.Spec = {
    FieldNames.PROMPT: lit_types.TextSegment(),
    FieldNames.TARGET: lit_types.TextSegment(required=False),
}

INPUT_SPEC_SALIENCE: lit_types.Spec = {
    FieldNames.TARGET_MASK: lit_types.TokenScores(align="", required=False),
}

OUTPUT_SPEC_GENERATION: lit_types.Spec = {
    FieldNames.RESPONSE: lit_types.GeneratedText(parent=FieldNames.TARGET)
}

OUTPUT_SPEC_GENERATION_EMBEDDINGS: lit_types.Spec = {
    FieldNames.PROMPT_EMBEDDINGS: lit_types.Embeddings(required=False),
    FieldNames.RESPONSE_EMBEDDINGS: lit_types.Embeddings(required=False),
}

OUTPUT_SPEC_TOKENIZER: lit_types.Spec = {
    FieldNames.TOKENS: lit_types.Tokens(parent=""),
}

OUTPUT_SPEC_SALIENCE: lit_types.Spec = {
    FieldNames.GRAD_DOT_INPUT: lit_types.TokenScores(align=FieldNames.TOKENS),
    FieldNames.GRAD_NORM: lit_types.TokenScores(align=FieldNames.TOKENS),
} | OUTPUT_SPEC_TOKENIZER
