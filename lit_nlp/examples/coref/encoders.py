"""Encoder implementation for frozen-encoder coref."""
import os
from typing import Dict

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.coref import retokenize
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils
import numpy as np
import tensorflow as tf
import transformers


class BertEncoderWithOffsets(lit_model.Model):
  """BERT encoder for pre-tokenized text."""

  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, model_name_or_path: str):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    self.model = model_utils.load_pretrained(
        transformers.TFBertForMaskedLM,
        model_name_or_path,
        output_hidden_states=True,
        output_attentions=False)

  def _postprocess(self, output: Dict[str, np.ndarray]):
    """Postprocess, modifying output dict in-place."""
    # Slice to remove padding, omitting initial [CLS] and final [SEP]
    slicer = slice(1, output.pop('ntok') - 1)
    output['wpm_tokens'] = self.tokenizer.convert_ids_to_tokens(
        output.pop('input_ids')[slicer])
    # <float>[num_tokens, emb_dim]
    output['top_layer_embs'] = output['top_layer_embs'][slicer]
    return output

  def save(self, path: str):
    if not os.path.isdir(path):
      os.mkdir(path)
    self.tokenizer.save_pretrained(path)  # configs and vocab files
    self.model.save_pretrained(path)  # as HDF5 weights file + config

  ##
  # LIT API implementations.
  def max_minibatch_size(self):
    # Rough heuristic for how much we can handle on one GPU with 12G of memory.
    return 64 if self.model.config.num_hidden_layers <= 12 else 32

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    tokens_and_offsets = [
        retokenize.subtokenize(ex['tokens'], self.tokenizer.tokenize)
        for ex in inputs
    ]
    tokenized_texts, offsets = zip(*tokens_and_offsets)
    # Process to ids, add special tokens, and compute segment ids and masks.
    encoded_input = self.tokenizer.batch_encode_plus(
        list(tokenized_texts),
        is_split_into_words=True,
        return_tensors='tf',
        add_special_tokens=True,
        max_length=self.max_seq_length,
        padding='longest',
        truncation='longest_first')

    out: transformers.modeling_tf_outputs.TFMaskedLMOutput = \
        self.model(encoded_input)
    batched_outputs = {
        'input_ids': encoded_input['input_ids'].numpy(),
        'ntok': tf.reduce_sum(encoded_input['attention_mask'], axis=1).numpy(),
        'top_layer_embs':
            out.hidden_states[-1].numpy(),  # last layer, all tokens
    }
    # List of dicts, one per example.
    unbatched_outputs = list(utils.unbatch_preds(batched_outputs))
    # Postprocess to remove padding and add offsets.
    ret = [self._postprocess(ubo) for ubo in unbatched_outputs]
    for preds, offset_indices in zip(ret, offsets):
      preds['offsets'] = offset_indices
    return ret

  def input_spec(self):
    return {'tokens': lit_types.Tokens()}

  def output_spec(self):
    return {
        'top_layer_embs':
            lit_types.TokenEmbeddings(),
        'wpm_tokens':
            lit_types.Tokens(),
        'offsets':
            lit_types.SubwordOffsets(align_in='tokens', align_out='wpm_tokens')
    }
