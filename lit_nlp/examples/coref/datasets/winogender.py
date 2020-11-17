"""Coreference version of the Winogender dataset.

Each instance has two edges, one between the pronoun and the occupation and one
between the pronoun and the participant. The pronoun is always span1.

There are 120 templates in the Winogender set, 60 coreferent with the
occupation, and 60 coreferent with the participant. Each is instantiated
six times: with and without "someone" substituting for the participant,
and with {male, female, neutral} pronouns, for a total of 720 examples.

Winogender repo: https://github.com/rudinger/winogender-schemas
Paper: Gender Bias in Coreference Resolution (Rudinger et al. 2018),
https://arxiv.org/pdf/1804.09301.pdf
"""
import enum
import os
from typing import Optional

from absl import logging
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import dtypes as lit_dtypes
from lit_nlp.api import types as lit_types
import pandas as pd
import transformers  # for file caching

EdgeLabel = lit_dtypes.EdgeLabel

DATA_ROOT = "https://raw.githubusercontent.com/rudinger/winogender-schemas/master/data/"  # pylint: disable=line-too-long


def get_data(name):
  """Download data or return local cache path."""
  url = os.path.join(DATA_ROOT, name)
  logging.info("Winogender: retrieving data file %s", url)
  return transformers.file_utils.cached_path(url)


## From gap-coreference/constants.py
class Gender(enum.Enum):
  UNKNOWN = 0
  MASCULINE = 1
  FEMININE = 2


NOM = "$NOM_PRONOUN"
POSS = "$POSS_PRONOUN"
ACC = "$ACC_PRONOUN"

PRONOUN_MAP = {
    Gender.FEMININE: {
        NOM: "she",
        POSS: "her",
        ACC: "her"
    },
    Gender.MASCULINE: {
        NOM: "he",
        POSS: "his",
        ACC: "him"
    },
    Gender.UNKNOWN: {
        NOM: "they",
        POSS: "their",
        ACC: "them"
    },
}

ANSWER_VOCAB = ["occupation", "participant"]

PRONOUNS_BY_GENDER = {k: "/".join(PRONOUN_MAP[k].values()) for k in PRONOUN_MAP}


# Based on winogender-schemas/scripts/instantiate.py, but adapted to LIT format.
def generate_instance(occupation,
                      participant,
                      answer,
                      sentence,
                      gender=Gender.UNKNOWN,
                      someone=False):
  """Generate a Winogender example from a template row."""
  toks = sentence.split(" ")
  part_index = toks.index("$PARTICIPANT")
  if not someone:
    # we are using the instantiated participant,
    # e.g. "client", "patient", "customer",...
    toks[part_index] = participant
  else:  # we are using the bleached NP "someone" for the other participant
    # first, remove the token that precedes $PARTICIPANT, i.e. "the"
    toks = toks[:part_index - 1] + toks[part_index:]
    # recompute participant index (it should be part_index - 1)
    part_index = toks.index("$PARTICIPANT")
    toks[part_index] = "Someone" if part_index == 0 else "someone"

  # Make sure we do this /after/ substituting "someone",
  # since that may change indices.
  occ_index = toks.index("$OCCUPATION")
  # This should always pass on the regular Winogender dataset.
  assert " " not in occupation, "Occupation must be single-token."
  toks[occ_index] = occupation

  pronoun_idx = None
  gendered_toks = []
  for i, t in enumerate(toks):
    sub = PRONOUN_MAP[gender].get(t, t)
    if sub != t:
      pronoun_idx = i
    gendered_toks.append(sub)

  # NOM, POSS, ACC
  pronoun_type = toks[pronoun_idx][1:].replace("_PRONOUN", "")

  # Process text for fluency
  text = " ".join(gendered_toks)
  text = text.replace("they was", "they were")
  text = text.replace("They was", "They were")

  record = {"text": text, "tokens": text.split()}
  t0 = EdgeLabel(
      span1=(occ_index, occ_index + 1),
      span2=(pronoun_idx, pronoun_idx + 1),
      label=int(1 if answer == 0 else 0))
  t1 = EdgeLabel(
      span1=(part_index, part_index + 1),
      span2=(pronoun_idx, pronoun_idx + 1),
      label=int(1 if answer == 1 else 0))
  record["coref"] = [t0, t1]
  record.update({
      "occupation": occupation,
      "participant": participant,
      "answer": ANSWER_VOCAB[answer],
      "someone": str(someone),
      "pronouns": PRONOUNS_BY_GENDER[gender],
      "pronoun_type": pronoun_type,
      "gender": gender.name,
  })
  return record


class WinogenderDataset(lit_dataset.Dataset):
  """Coreference on Winogender schemas (Rudinger et al. 2018)."""

  # These should match the args to generate_instance()
  TSV_COLUMN_NAMES = ["occupation", "participant", "answer", "sentence"]

  def __init__(self,
               templates_path: Optional[str] = None,
               occupation_stats_path: Optional[str] = None):
    templates_path = templates_path or get_data("templates.tsv")
    occupation_stats_path = occupation_stats_path or get_data(
        "occupations-stats.tsv")

    # Load templates and make a DataFrame.
    with open(templates_path) as fd:
      self.templates_df = pd.read_csv(
          fd, sep="\t", header=0, names=self.TSV_COLUMN_NAMES)

    # Load occpuation stats.
    with open(occupation_stats_path) as fd:
      self.occupation_df = pd.read_csv(fd, sep="\t").set_index("occupation")

    # Make examples for each {someone} x {gender} x {template}
    self._examples = []
    for _, row in self.templates_df.iterrows():
      for someone in {False, True}:
        for gender in Gender:
          r = generate_instance(someone=someone, gender=gender, **row)
          r["pf_bls"] = (
              self.occupation_df.bls_pct_female[r["occupation"]] / 100.0)
          self._examples.append(r)

  def spec(self):
    return {
        "text":
            lit_types.TextSegment(),
        "tokens":
            lit_types.Tokens(parent="text"),
        "coref":
            lit_types.EdgeLabels(align="tokens"),
        # Metadata fields for filtering and analysis.
        "occupation":
            lit_types.CategoryLabel(),
        "participant":
            lit_types.CategoryLabel(),
        "answer":
            lit_types.CategoryLabel(vocab=ANSWER_VOCAB),
        "someone":
            lit_types.CategoryLabel(vocab=["True", "False"]),
        "pronouns":
            lit_types.CategoryLabel(vocab=list(PRONOUNS_BY_GENDER.values())),
        "pronoun_type":
            lit_types.CategoryLabel(vocab=["NOM", "POSS", "ACC"]),
        "gender":
            lit_types.CategoryLabel(vocab=[g.name for g in Gender]),
        "pf_bls":
            lit_types.Scalar(),
    }
