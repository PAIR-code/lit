"""Tests for types."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from lit_nlp.api import dtypes
from lit_nlp.api import types
import numpy as np


class TypesTest(parameterized.TestCase):

  def test_inherit_parent_default_type(self):
    lit_type = types.StringLitType()
    self.assertIsInstance(lit_type.default, str)

  def test_inherit_parent_default_value(self):
    lit_type = types.SingleFieldMatcher(spec="dataset", types=["LitType"])
    self.assertIsNone(lit_type.default)

  def test_requires_parent_custom_properties(self):
    # TokenSalience requires the `signed` property of its parent class.
    with self.assertRaises(TypeError):
      _ = types.TokenSalience(autorun=True)

  def test_inherit_parent_custom_properties(self):
    lit_type = types.TokenSalience(autorun=True, signed=True)
    self.assertIsNone(lit_type.default)

    lit_type = types.TokenGradients(
        grad_for="cls_emb", grad_target_field_key="grad_class")
    self.assertTrue(hasattr(lit_type, "align"))
    self.assertFalse(hasattr(lit_type, "not_a_property"))

  @parameterized.named_parameters(
      ("list[int]", [1, 2, 3], 1),
      ("np_array[int]", np.array([1, 2, 3]), 1),
      ("np_array[list[int]]", np.array([[1, 1], [2, 3]]), 2),
      ("np_array[list[int]]_2_dim", np.array([[1, 1], [2, 3]]), [2, 4]),
  )
  def test_tensor_ndim(self, value, ndim):
    emb = types.Embeddings()
    try:
      emb.validate_ndim(value, ndim)
    except ValueError:
      self.fail("Raised unexpected error.")

  @parameterized.named_parameters(
      ("ndim_wrong_size", [1, 2, 3], 2),
      ("ndim_wrong_type", np.array([[1, 1], [2, 3]]), [1]),
  )
  def test_tensor_ndim_errors(self, value, ndim):
    with self.assertRaises(ValueError):
      emb = types.Embeddings()
      emb.validate_ndim(value, ndim)

  @parameterized.named_parameters(
      ("boolean", types.Boolean(), True),
      ("embeddings_list[int]", types.Embeddings(), [1, 2]),
      ("embeddings_np_array", types.Embeddings(), np.array([1, 2])),
      ("image", types.ImageBytes(), "data:image/blah..."),
      ("scalar_float", types.Scalar(), 3.4),
      ("scalar_int", types.Scalar(), 3),
      ("scalar_numpy", types.Scalar(), np.int64(2)),
      ("text", types.TextSegment(), "hi"),
      ("tokens", types.Tokens(), ["a", "b"]),
  )
  def test_type_validate_input(self, lit_type: types.LitType, value: Any):
    spec = {"score": types.Scalar(), "text": types.TextSegment()}
    example = {}
    try:
      lit_type.validate_input(value, spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

  @parameterized.named_parameters(
      ("boolean_number", types.Boolean(), 3.14159),
      ("boolean_text", types.Boolean(), "hi"),
      ("embeddings_bool", types.Embeddings(), True),
      ("embeddings_number", types.Embeddings(), 3.14159),
      ("embeddings_text", types.Embeddings(), "hi"),
      ("image_bool", types.ImageBytes(), True),
      ("image_number", types.ImageBytes(), 3.14159),
      ("image_text", types.ImageBytes(), "hi"),
      ("scalar_text", types.Scalar(), "hi"),
      ("text_bool", types.TextSegment(), True),
      ("text_number", types.TextSegment(), 3.14159),
      ("tokens_bool", types.Tokens(), True),
      ("tokens_number", types.Tokens(), 3.14159),
      ("tokens_text", types.Tokens(), "hi"),
  )
  def test_type_validate_input_errors(self,
                                      lit_type: types.LitType,
                                      value: Any):
    spec = {"score": types.Scalar(), "text": types.TextSegment()}
    example = {}
    with self.assertRaises(ValueError):
      lit_type.validate_input(value, spec, example)

  @parameterized.named_parameters(
      dict(
          testcase_name="CategoryLabel",
          json_dict={
              "required": False,
              "annotated": False,
              "default": "",
              "vocab": ["0", "1"],
              "__name__": "CategoryLabel",
          },
          expected_type=types.CategoryLabel,
      ),
      dict(
          testcase_name="Embeddings",
          json_dict={
              "required": True,
              "annotated": False,
              "default": None,
              "__name__": "Embeddings",
          },
          expected_type=types.Embeddings,
      ),
      dict(
          testcase_name="Gradients",
          json_dict={
              "required": True,
              "annotated": False,
              "default": None,
              "align": None,
              "grad_for": "cls_emb",
              "grad_target_field_key": "grad_class",
              "__name__": "Gradients",
          },
          expected_type=types.Gradients,
      ),
      dict(
          testcase_name="MulticlassPreds",
          json_dict={
              "required": True,
              "annotated": False,
              "default": None,
              "vocab": ["0", "1"],
              "null_idx": 0,
              "parent": "label",
              "autosort": False,
              "threshold": None,
              "__name__": "MulticlassPreds",
          },
          expected_type=types.MulticlassPreds,
      ),
      dict(
          testcase_name="RegressionScore",
          json_dict={
              "required": True,
              "annotated": False,
              "min_val": 0,
              "max_val": 1,
              "default": 0,
              "step": 0.01,
              "parent": "label",
              "__name__": "RegressionScore",
          },
          expected_type=types.RegressionScore,
      ),
      dict(
          testcase_name="Scalar",
          json_dict={
              "required": True,
              "annotated": False,
              "min_val": 2,
              "max_val": 100,
              "default": 10,
              "step": 1,
              "__name__": "Scalar",
          },
          expected_type=types.Scalar,
      ),
      dict(
          testcase_name="TextSegment",
          json_dict={
              "required": True,
              "annotated": False,
              "default": "",
              "__name__": "TextSegment",
          },
          expected_type=types.TextSegment,
      ),
      dict(
          testcase_name="TokenEmbeddings",
          json_dict={
              "required": True,
              "annotated": False,
              "default": None,
              "align": "tokens_sentence",
              "__name__": "TokenEmbeddings",
          },
          expected_type=types.TokenEmbeddings,
      ),
      dict(
          testcase_name="Tokens",
          json_dict={
              "required": False,
              "annotated": False,
              "default": [],
              "parent": "sentence",
              "mask_token": None,
              "token_prefix": "##",
              "__name__": "Tokens",
          },
          expected_type=types.Tokens,
      ),
  )
  def test_from_json(self, json_dict: types.JsonDict,
                     expected_type: types.LitType):
    lit_type: types.LitType = types.LitType.from_json(json_dict)
    self.assertIsInstance(lit_type, expected_type)
    for key in json_dict:
      if key == "__name__":
        continue
      elif hasattr(lit_type, key):
        self.assertEqual(getattr(lit_type, key), json_dict[key])
      else:
        self.fail(f"Encountered unknown property {key} for type "
                  f"{lit_type.__class__.__name__}.")

  @parameterized.named_parameters(
      ("empty_dict", {}, KeyError),
      ("invalid_name_empty", {"__name__": ""}, NameError),
      ("invalid_name_none", {"__name__": None}, TypeError),
      ("invalid_name_number", {"__name__": 3.14159}, TypeError),
      ("invalid_type_name", {"__name__": "not_a_lit_type"}, NameError),
  )
  def test_from_json_errors(self, value: types.JsonDict, expected_error):
    with self.assertRaises(expected_error):
      _ = types.LitType.from_json(value)

  def test_type_validate_gentext_output(self):
    ds_spec = {
        "num": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "gentext": types.GeneratedText(parent="text"),
        "cands": types.GeneratedTextCandidates(parent="text")
    }
    example = {"num": 1, "text": "hi"}
    output = {"gentext": "test", "cands": [("hi", 4), ("bye", None)]}

    gentext = types.GeneratedText(parent="text")
    gentextcands = types.GeneratedTextCandidates(parent="text")
    try:
      gentext.validate_output("hi", out_spec, output, ds_spec, ds_spec, example)
      gentextcands.validate_output([("hi", 4), ("bye", None)], out_spec, output,
                                   ds_spec, ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    bad_gentext = types.GeneratedText(parent="num")
    self.assertRaises(ValueError, bad_gentext.validate_output, "hi", out_spec,
                      output, ds_spec, ds_spec, example)

    self.assertRaises(ValueError, gentextcands.validate_output,
                      [("hi", "wrong"), ("bye", None)], out_spec, output,
                      ds_spec, ds_spec, example)
    bad_gentextcands = types.GeneratedTextCandidates(parent="num")
    self.assertRaises(ValueError, bad_gentextcands.validate_output,
                      [("hi", 4), ("bye", None)], out_spec, output, ds_spec,
                      ds_spec, example)

  def test_type_validate_genurl(self):
    ds_spec = {
        "text": types.TextSegment(),
    }
    out_spec = {
        "genurl": types.GeneratedURL(align="cands"),
        "cands": types.GeneratedTextCandidates(parent="text")
    }
    example = {"text": "hi"}
    output = {"genurl": "https://blah", "cands": [("hi", 4), ("bye", None)]}

    genurl = types.GeneratedURL(align="cands")
    try:
      genurl.validate_output("https://blah", out_spec, output, ds_spec, ds_spec,
                             example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, genurl.validate_output, 4,
                      out_spec, output, ds_spec, ds_spec, example)
    bad_genurl = types.GeneratedURL(align="wrong")
    self.assertRaises(ValueError, bad_genurl.validate_output, "https://blah",
                      out_spec, output, ds_spec, ds_spec, example)

  def test_tokentopk(self):
    ds_spec = {
        "text": types.TextSegment(),
    }
    out_spec = {
        "tokens": types.Tokens(),
        "preds": types.TokenTopKPreds(align="tokens")
    }
    example = {"text": "hi"}
    output = {"tokens": ["hi"], "preds": [[("one", .9), ("two", .4)]]}

    preds = types.TokenTopKPreds(align="tokens")
    try:
      preds.validate_output(
          [[("one", .9), ("two", .4)]], out_spec, output, ds_spec, ds_spec,
          example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(
        ValueError, preds.validate_output,
        [[("one", .2), ("two", .4)]], out_spec, output, ds_spec, ds_spec,
        example)
    self.assertRaises(
        ValueError, preds.validate_output,
        [["one", "two"]], out_spec, output, ds_spec, ds_spec, example)
    self.assertRaises(
        ValueError, preds.validate_output, ["wrong"], out_spec, output,
        ds_spec, ds_spec, example)

    bad_preds = types.TokenTopKPreds(align="preds")
    self.assertRaises(
        ValueError, bad_preds.validate_output,
        [[("one", .9), ("two", .4)]], out_spec, output, ds_spec, ds_spec,
        example)

  def test_regression(self):
    ds_spec = {
        "val": types.Scalar(),
        "text": types.TextSegment(),
    }
    out_spec = {
        "score": types.RegressionScore(parent="val"),
    }
    example = {"val": 2}
    output = {"score": 1}

    score = types.RegressionScore(parent="val")
    try:
      score.validate_output(1, out_spec, output, ds_spec, ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, score.validate_output, "wrong",
                      out_spec, output, ds_spec, ds_spec, example)
    bad_score = types.RegressionScore(parent="text")
    self.assertRaises(ValueError, bad_score.validate_output, 1,
                      out_spec, output, ds_spec, ds_spec, example)

  def test_reference(self):
    ds_spec = {
        "text": types.TextSegment(),
        "val": types.Scalar(),
    }
    out_spec = {
        "scores": types.ReferenceScores(parent="text"),
    }
    example = {"text": "hi"}
    output = {"scores": [1, 2]}

    score = types.ReferenceScores(parent="text")
    try:
      score.validate_output([1, 2], out_spec, output, ds_spec, ds_spec, example)
      score.validate_output(np.array([1, 2]), out_spec, output, ds_spec,
                            ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, score.validate_output, ["a"],
                      out_spec, output, ds_spec, ds_spec, example)
    bad_score = types.ReferenceScores(parent="val")
    self.assertRaises(ValueError, bad_score.validate_output, [1],
                      out_spec, output, ds_spec, ds_spec, example)

  def test_multiclasspreds(self):
    ds_spec = {
        "label": types.CategoryLabel(),
        "val": types.Scalar(),
    }
    out_spec = {
        "scores": types.MulticlassPreds(
            parent="label", null_idx=0, vocab=["a", "b"]),
    }
    example = {"label": "hi", "val": 1}
    output = {"scores": [1, 2]}

    score = types.MulticlassPreds(parent="label", null_idx=0, vocab=["a", "b"])
    try:
      score.validate_output([1, 2], out_spec, output, ds_spec, ds_spec, example)
      score.validate_output(np.array([1, 2]), out_spec, output, ds_spec,
                            ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, score.validate_output, ["a", "b"],
                      out_spec, output, ds_spec, ds_spec, example)
    bad_score = types.MulticlassPreds(
        parent="label", null_idx=2, vocab=["a", "b"])
    self.assertRaises(ValueError, bad_score.validate_output, [1, 2],
                      out_spec, output, ds_spec, ds_spec, example)
    bad_score = types.MulticlassPreds(
        parent="val", null_idx=0, vocab=["a", "b"])
    self.assertRaises(ValueError, bad_score.validate_output, [1, 2],
                      out_spec, output, ds_spec, ds_spec, example)

  def test_annotations(self):
    ds_spec = {
        "text": types.TextSegment(),
    }
    out_spec = {
        "tokens": types.Tokens(),
        "spans": types.SpanLabels(align="tokens"),
        "edges": types.EdgeLabels(align="tokens"),
        "annot": types.MultiSegmentAnnotations(),
    }
    example = {"text": "hi"}
    output = {"tokens": ["hi"], "preds": [dtypes.SpanLabel(start=0, end=1)],
              "edges": [dtypes.EdgeLabel(span1=(0, 0), span2=(1, 1), label=0)],
              "annot": [dtypes.AnnotationCluster(label="hi", spans=[])]}

    spans = types.SpanLabels(align="tokens")
    edges = types.EdgeLabels(align="tokens")
    annot = types.MultiSegmentAnnotations()
    try:
      spans.validate_output(
          [dtypes.SpanLabel(start=0, end=1)], out_spec, output, ds_spec,
          ds_spec, example)
      edges.validate_output(
          [dtypes.EdgeLabel(span1=(0, 0), span2=(1, 1), label=0)], out_spec,
          output, ds_spec, ds_spec, example)
      annot.validate_output(
          [dtypes.AnnotationCluster(label="hi", spans=[])], out_spec,
          output, ds_spec, ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(
        ValueError, spans.validate_output, [1], out_spec, output, ds_spec,
        ds_spec, example)
    self.assertRaises(
        ValueError, edges.validate_output, [1], out_spec, output, ds_spec,
        ds_spec, example)
    self.assertRaises(
        ValueError, annot.validate_output, [1], out_spec, output, ds_spec,
        ds_spec, example)

    bad_spans = types.SpanLabels(align="edges")
    bad_edges = types.EdgeLabels(align="spans")
    self.assertRaises(
        ValueError, bad_spans.validate_output,
        [dtypes.SpanLabel(start=0, end=1)], out_spec, output, ds_spec, ds_spec,
        example)
    self.assertRaises(
        ValueError, bad_edges.validate_output,
        [dtypes.EdgeLabel(span1=(0, 0), span2=(1, 1), label=0)], out_spec,
        output, ds_spec, ds_spec, example)

  def test_gradients(self):
    ds_spec = {
        "text": types.TextSegment(),
        "target": types.CategoryLabel()
    }
    out_spec = {
        "tokens": types.Tokens(),
        "embs": types.Embeddings(),
        "grads": types.Gradients(align="tokens", grad_for="embs",
                                 grad_target_field_key="target")
    }
    example = {"text": "hi", "target": "one"}
    output = {"tokens": ["hi"], "embs": [.1, .2], "grads": [.1]}

    grads = types.Gradients(align="tokens", grad_for="embs",
                            grad_target_field_key="target")
    embs = types.Embeddings()
    try:
      grads.validate_output([.1], out_spec, output, ds_spec, ds_spec, example)
      embs.validate_output([.1, .2], out_spec, output, ds_spec, ds_spec,
                           example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(
        ValueError, grads.validate_output, ["bad"], out_spec, output, ds_spec,
        ds_spec, example)
    self.assertRaises(
        ValueError, embs.validate_output, ["bad"], out_spec, output, ds_spec,
        ds_spec, example)

    bad_grads = types.Gradients(align="text", grad_for="embs",
                                grad_target_field_key="target")
    self.assertRaises(
        ValueError, bad_grads.validate_output, [.1], out_spec, output, ds_spec,
        ds_spec, example)
    bad_grads = types.Gradients(align="tokens", grad_for="tokens",
                                grad_target_field_key="target")
    self.assertRaises(
        ValueError, bad_grads.validate_output, [.1], out_spec, output, ds_spec,
        ds_spec, example)
    bad_grads = types.Gradients(align="tokens", grad_for="embs",
                                grad_target_field_key="bad")
    self.assertRaises(
        ValueError, bad_grads.validate_output, [.1], out_spec, output, ds_spec,
        ds_spec, example)

  def test_tokenembsgrads(self):
    ds_spec = {
        "text": types.TextSegment(),
        "target": types.CategoryLabel()
    }
    out_spec = {
        "tokens": types.Tokens(),
        "embs": types.TokenEmbeddings(align="tokens"),
        "grads": types.TokenGradients(align="tokens", grad_for="embs",
                                      grad_target_field_key="target")
    }
    example = {"text": "hi", "target": "one"}
    output = {"tokens": ["hi"], "embs": np.array([[.1], [.2]]),
              "grads": np.array([[.1], [.2]])}

    grads = types.TokenGradients(align="tokens", grad_for="embs",
                                 grad_target_field_key="target")
    embs = types.TokenEmbeddings(align="tokens")
    try:
      grads.validate_output(np.array([[.1], [.2]]), out_spec, output, ds_spec,
                            ds_spec, example)
      embs.validate_output(np.array([[.1], [.2]]), out_spec, output, ds_spec,
                           ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(
        ValueError, grads.validate_output, np.array([.1, .2]), out_spec, output,
        ds_spec, ds_spec, example)
    self.assertRaises(
        ValueError, embs.validate_output, np.array([.1, .2]), out_spec, output,
        ds_spec, ds_spec, example)

    bad_embs = types.TokenEmbeddings(align="grads")
    self.assertRaises(
        ValueError, bad_embs.validate_output, np.array([[.1], [.2]]), out_spec,
        output, ds_spec, ds_spec, example)

  def test_attention(self):
    ds_spec = {
        "text": types.TextSegment(),
    }
    out_spec = {
        "tokens": types.Tokens(),
        "val": types.RegressionScore,
        "attn": types.AttentionHeads(align_in="tokens", align_out="tokens"),
    }
    example = {"text": "hi"}
    output = {"tokens": ["hi"], "attn": np.array([[[.1]], [[.2]]])}

    attn = types.AttentionHeads(align_in="tokens", align_out="tokens")
    try:
      attn.validate_output(np.array([[[.1]], [[.2]]]), out_spec, output,
                           ds_spec, ds_spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(
        ValueError, attn.validate_output, np.array([.1, .2]), out_spec, output,
        ds_spec, ds_spec, example)

    bad_attn = types.AttentionHeads(align_in="tokens", align_out="val")
    self.assertRaises(
        ValueError, bad_attn.validate_output, np.array([[[.1]], [[.2]]]),
        out_spec, output, ds_spec, ds_spec, example)
    bad_attn = types.AttentionHeads(align_in="val", align_out="tokens")
    self.assertRaises(
        ValueError, bad_attn.validate_output, np.array([[[.1]], [[.2]]]),
        out_spec, output, ds_spec, ds_spec, example)


if __name__ == "__main__":
  absltest.main()
