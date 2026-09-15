"""
Extracts biomedical entities (findings, anatomy, conditions) from radiology
report text using scispaCy, a spaCy pipeline trained on biomedical text.

Setup:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

    Note: we're using the base en_core_sci_sm model, which only needs the
    two commands above. A more advanced setup exists (UMLS entity linking)
    but that requires an extra dependency called nmslib which can be tricky
    to install on Windows -- skip that for now, this base model is enough
    to get real entities out of the text and into the knowledge graph.

Usage:
    from src.knowledge_graph.ner_extractor import NERExtractor
    extractor = NERExtractor()
    entities = extractor.extract("There is no focal consolidation or pleural effusion.")
"""

import re

import spacy


class NERExtractor:
    def __init__(self, model_name: str = "en_core_sci_sm"):
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> list:
        """
        Returns a list of entity strings found in the text.
        e.g. "no focal consolidation" -> ["focal consolidation"]
        """
        if not text or not isinstance(text, str):
            return []

        # the IU X-ray dataset masks identifying info with "XXXX" -- strip
        # it out so it doesn't get picked up or interfere with entity spans
        cleaned = re.sub(r"\bXXXX\b", "", text)

        doc = self.nlp(cleaned)
        entities = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
        return entities

    def extract_with_negation(self, text: str) -> list:
        """
        Basic negation-aware extraction: flags whether an entity appears in
        a negated context (e.g. "no pneumothorax"). This matters a lot for
        radiology reports, where most sentences describe the ABSENCE of a
        finding, not its presence -- getting this wrong would make the
        knowledge graph claim patients have conditions they were actually
        cleared of.

        Returns a list of dicts: {"entity": str, "negated": bool}
        """
        if not text or not isinstance(text, str):
            return []

        cleaned = re.sub(r"\bXXXX\b", "", text)
        doc = self.nlp(cleaned)

        negation_cues = {"no", "not", "without", "negative", "free", "denies", "absent"}

        results = []
        for ent in doc.ents:
            # look at up to 4 tokens before the entity for a negation cue
            window_start = max(0, ent.start - 4)
            preceding_tokens = {t.text.lower() for t in doc[window_start:ent.start]}
            negated = bool(preceding_tokens & negation_cues)
            results.append({"entity": ent.text.strip(), "negated": negated})

        return results


if __name__ == "__main__":
    extractor = NERExtractor()

    sample_texts = [
        "The cardiac silhouette and mediastinum size are within normal limits. "
        "There is no pulmonary edema. There is no focal consolidation. "
        "There are no XXXX of a pleural effusion.",
        "Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries.",
    ]

    for text in sample_texts:
        print("=" * 60)
        print("Text:", text)
        print("Entities:", extractor.extract(text))
        print("With negation:")
        for r in extractor.extract_with_negation(text):
            flag = "NEGATED" if r["negated"] else "present"
            print(f"  - {r['entity']}  [{flag}]")