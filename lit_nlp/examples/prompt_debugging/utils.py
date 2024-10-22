"""Utility functions for Prompt Debugging use cases."""


def generate_model_group_names(name: str) -> tuple[str, str]:
  return f"_{name}_salience", f"_{name}_tokenizer"
