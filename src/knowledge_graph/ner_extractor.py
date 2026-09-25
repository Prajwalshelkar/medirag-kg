"""
Extracts biomedical entities (findings, anatomy, conditions) from radiology
report text using scispaCy, with negation detection handled by negspacy
(an established spaCy extension implementing the NegEx algorithm).

Why negspacy instead of a hand-rolled word-window check: radiology reports
often list multiple negated findings in one clause, e.g. "no rib fractures,
pneumothorax, or pleural effusion identified" -- a simple "look N words
back for a negation cue" approach fails on the later items in that list,
because the negation word is too far back. negspacy scopes negation
properly across the whole clause instead of a fixed word window, so it
correctly catches all three items as negated, not just the first.

Setup:
    pip install scispacy negspacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

Usage:
    from src.knowledge_graph.ner_extractor import NERExtractor
    extractor = NERExtractor()
    entities = extractor.extract_with_negation("There is no focal consolidation.")
"""

import re

import spacy
from negspacy.negation import Negex  # noqa: F401 -- registers the "negex" pipe


class NERExtractor:
    def __init__(self, model_name: str = "en_core_sci_sm"):
        self.nlp = spacy.load(model_name)

        if "negex" not in self.nlp.pipe_names:
            # ent_types=[] means: apply negation detection to ALL entity
            # types this model produces (en_core_sci_sm's generic NER
            # doesn't use fine-grained type labels, so we don't restrict it)
            self.nlp.add_pipe("negex", config={"ent_types": []})

    def _clean(self, text: str) -> str:
        # the IU X-ray dataset masks identifying info with "XXXX" -- strip
        # it out so it doesn't get picked up or interfere with entity spans
        return re.sub(r"\bXXXX\b", "", text) if isinstance(text, str) else ""

    def extract(self, text: str) -> list:
        """Returns a list of entity strings found in the text."""
        if not text:
            return []
        doc = self.nlp(self._clean(text))
        return [ent.text.strip() for ent in doc.ents if ent.text.strip()]

    def extract_with_negation(self, text: str) -> list:
        """
        Returns a list of dicts: {"entity": str, "negated": bool}
        Negation is determined by negspacy's NegEx-based clause scoping,
        which correctly handles lists like "no A, B, or C" -- not just the
        first item after the negation word.
        """
        if not text:
            return []
        doc = self.nlp(self._clean(text))
        return [
            {"entity": ent.text.strip(), "negated": bool(ent._.negex)}
            for ent in doc.ents
            if ent.text.strip()
        ]


if __name__ == "__main__":
    extractor = NERExtractor()

    sample_texts = [
        "The cardiac silhouette and mediastinum size are within normal limits. "
        "There is no pulmonary edema. There is no focal consolidation. "
        "There are no XXXX of a pleural effusion.",
        "Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries.",
        # the exact case that broke the old word-window approach:
        "No displaced rib fractures, pneumothorax, or pleural effusion identified. "
        "Well-expanded and clear lungs.",
    ]

    for text in sample_texts:
        print("=" * 60)
        print("Text:", text)
        for r in extractor.extract_with_negation(text):
            flag = "NEGATED" if r["negated"] else "present"
            print(f"  - {r['entity']}  [{flag}]")