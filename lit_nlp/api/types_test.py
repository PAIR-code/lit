"""Tests for types."""

from absl.testing import absltest
from lit_nlp.api import dtypes
from lit_nlp.api import types
import numpy as np


class TypesTest(absltest.TestCase):

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

  def test_tensor_ndim(self):
    emb = types.Embeddings()
    try:
      emb.validate_ndim([1, 2, 3], 1)
      emb.validate_ndim(np.array([1, 2, 3]), 1)
      emb.validate_ndim(np.array([[1, 1], [2, 3]]), 2)
      emb.validate_ndim(np.array([[1, 1], [2, 3]]), [2, 4])
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, emb.validate_ndim, [1, 2, 3], 2)
    self.assertRaises(ValueError, emb.validate_ndim, np.array([[1, 1], [2, 3]]),
                      [1])

  def test_type_validate_input(self):
    spec = {
        "score": types.Scalar(),
        "text": types.TextSegment(),
    }
    example = {}
    scalar = types.Scalar()
    text = types.TextSegment()
    img = types.ImageBytes()
    tok = types.Tokens()
    emb = types.Embeddings()
    bl = types.Boolean()
    try:
      scalar.validate_input(3.4, spec, example)
      scalar.validate_input(3, spec, example)
      scalar.validate_input(np.int64(2), spec, example)
      text.validate_input("hi", spec, example)
      img.validate_input("data:image/blah...", spec, example)
      tok.validate_input(["a", "b"], spec, example)
      emb.validate_input([1, 2], spec, example)
      emb.validate_input(np.array([1, 2]), spec, example)
      bl.validate_input(True, spec, example)
    except ValueError:
      self.fail("Raised unexpected error.")

    self.assertRaises(ValueError, scalar.validate_input, "hi", spec, example)
    self.assertRaises(ValueError, img.validate_input, "hi", spec, example)
    self.assertRaises(ValueError, text.validate_input, 4, spec, example)
    self.assertRaises(ValueError, tok.validate_input, [1], spec, example)
    self.assertRaises(ValueError, emb.validate_input, ["a"], spec, example)
    self.assertRaises(ValueError, bl.validate_input, 4, spec, example)

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
