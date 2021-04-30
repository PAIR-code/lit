"""Tests for model_utils."""
import os

from absl.testing import absltest
from lit_nlp.examples.models import model_utils
import numpy as np
import transformers

TESTDATA_PATH = os.path.join(os.path.dirname(__file__), 'testdata')


class BatchEncodePretokenizedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tokenizer_path = os.path.join(TESTDATA_PATH, 'bert_tokenizer')
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

  def test_tokenizer(self):
    sentences = [
        'Hello World!', 'Pineapple is not in the BERT vocabulary.', 'foobar',
        'spam and eggs'
    ]
    expected_tokens = [['hello', 'world', '!'],
                       [
                           'pine', '##apple', 'is', 'not', 'in', 'the', 'bert',
                           'vocabulary', '.'
                       ], ['foo', '##bar'], ['spa', '##m', 'and', 'eggs']]
    tokens = [self.tokenizer.tokenize(s) for s in sentences]
    self.assertEqual(tokens, expected_tokens)

  def test_tokenized_single(self):
    tokens = [['hello', 'world', '!'],
              [
                  'pine', '##apple', 'is', 'not', 'in', 'the', 'bert',
                  'vocabulary', '.'
              ]]
    encoded_input = model_utils.batch_encode_pretokenized(
        self.tokenizer, tokens)
    # We don't care about the raw IDs, but let's detokenize and make sure we get
    # something reasonable.
    recovered_tokens = [
        self.tokenizer.convert_ids_to_tokens(ids[:ntok]) for ids, ntok in zip(
            encoded_input['input_ids'].numpy(),
            encoded_input['attention_mask'].numpy().sum(axis=1))
    ]
    expected_tokens = [['[CLS]'] + ts + ['[SEP]'] for ts in tokens]
    self.assertEqual(recovered_tokens, expected_tokens)

  def test_tokenized_pairs(self):
    input_pairs = [[['hello', 'world', '!'], ['pine', '##apple']],
                   [['foo', '##bar'], ['spa', '##m', 'and', 'eggs']]]
    encoded_input = model_utils.batch_encode_pretokenized(
        self.tokenizer, *zip(*input_pairs))
    # We don't care about the raw IDs, but let's detokenize and make sure we get
    # something reasonable.
    recovered_tokens = [
        self.tokenizer.convert_ids_to_tokens(ids[:ntok]) for ids, ntok in zip(
            encoded_input['input_ids'].numpy(),
            encoded_input['attention_mask'].numpy().sum(axis=1))
    ]
    expected_tokens = [
        ['[CLS]'] + ip[0] + ['[SEP]'] + ip[1] + ['[SEP]'] for ip in input_pairs
    ]
    self.assertEqual(recovered_tokens, expected_tokens)

  def test_untokenized_single(self):
    """Check that this matches batch_encode_plus on the original text."""
    sentences = [
        'Hello World!', 'Pineapple is not in the BERT vocabulary.', 'foobar',
        'spam and eggs'
    ]
    tokens = [self.tokenizer.tokenize(s) for s in sentences]
    encoded = model_utils.batch_encode_pretokenized(self.tokenizer, tokens)
    expected_encoded = self.tokenizer.batch_encode_plus(
        sentences,
        return_tensors='tf',
        add_special_tokens=True,
        padding='longest',
        truncation='longest_first')
    for key in expected_encoded.keys():
      np.testing.assert_array_equal(encoded[key].numpy(),
                                    expected_encoded[key].numpy())

  def test_untokenized_pairs(self):
    """Check that this matches batch_encode_plus on the original text."""
    sentence_pairs = [('Hello World!',
                       'Pineapple is not in the BERT vocabulary.'),
                      ('foobar', 'spam and eggs')]
    input_tokens = [
        [self.tokenizer.tokenize(s) for s in pair] for pair in sentence_pairs
    ]
    encoded = model_utils.batch_encode_pretokenized(self.tokenizer,
                                                    *zip(*input_tokens))
    expected_encoded = self.tokenizer.batch_encode_plus(
        sentence_pairs,
        return_tensors='tf',
        add_special_tokens=True,
        padding='longest',
        truncation='longest_first')
    for key in expected_encoded.keys():
      np.testing.assert_array_equal(encoded[key].numpy(),
                                    expected_encoded[key].numpy())

  def test_extra_kw(self):
    """Check that this matches batch_encode_plus on the original text."""
    kw = dict(max_length=6)
    sentences = ['This is a somewhat longer sentence that should be truncated.']
    tokens = [self.tokenizer.tokenize(s) for s in sentences]
    encoded = model_utils.batch_encode_pretokenized(self.tokenizer, tokens,
                                                    **kw)
    expected_encoded = self.tokenizer.batch_encode_plus(
        sentences,
        return_tensors='tf',
        add_special_tokens=True,
        padding='longest',
        truncation='longest_first',
        **kw)
    for key in expected_encoded.keys():
      np.testing.assert_array_equal(encoded[key].numpy(),
                                    expected_encoded[key].numpy())


if __name__ == '__main__':
  absltest.main()
