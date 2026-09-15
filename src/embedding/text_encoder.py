"""
Encodes radiology report text into vector embeddings for the RAG retrieval
side (separate from ner_extractor.py, which pulls out entities for the
knowledge graph -- this is for semantic similarity search over report text).

Uses a sentence-transformers model pretrained on biomedical text and tuned
for search/retrieval tasks (trained on MS-MARCO, a passage-retrieval
dataset), which matters because retrieval needs "does this text answer the
query" similarity, not just general topic similarity.

Setup:
    pip install sentence-transformers

Usage:
    from src.embedding.text_encoder import TextEncoder
    encoder = TextEncoder()
    vector = encoder.encode("There is no focal consolidation or pleural effusion.")
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """Returns a 1D embedding vector for one piece of report text."""
        if not text or not isinstance(text, str):
            text = ""
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list) -> np.ndarray:
        """Encodes multiple texts at once. Returns shape (N, embedding_dim)."""
        cleaned = [t if isinstance(t, str) else "" for t in texts]
        return self.model.encode(cleaned, convert_to_numpy=True, show_progress_bar=True)


if __name__ == "__main__":
    encoder = TextEncoder(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

    sample_text = (
        "The cardiac silhouette and mediastinum size are within normal limits. "
        "There is no pulmonary edema. There is no focal consolidation."
    )

    vec = encoder.encode(sample_text)
    print(f"Embedding shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")

    # quick sanity check: similar reports should have high cosine similarity,
    # unrelated ones should have lower similarity
    other_text = "No acute cardiopulmonary abnormality. Clear lungs bilaterally."
    unrelated_text = "Patient reports history of chronic obstructive lung disease."

    vec2 = encoder.encode(other_text)
    vec3 = encoder.encode(unrelated_text)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nSimilarity (normal vs normal-ish): {cosine_sim(vec, vec2):.4f}")
    print(f"Similarity (normal vs COPD mention): {cosine_sim(vec, vec3):.4f}")