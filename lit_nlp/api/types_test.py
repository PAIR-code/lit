"""Tests for types."""

from lit_nlp.api import types
from google3.testing.pybase import googletest


class TypesTest(googletest.TestCase):

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


if __name__ == "__main__":
  googletest.main()
