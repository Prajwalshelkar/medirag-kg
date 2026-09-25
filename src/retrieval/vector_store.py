"""
Stores image and text embeddings in ChromaDB (a free, local vector
database) and provides similarity search over them.

Two separate collections are kept: one for image embeddings, one for
text embeddings, since they come from different encoders with different
vector spaces. Each stored vector is tagged with metadata (report uid,
image filename, etc.) so a similarity search result can be traced back
to the actual report/image it came from.

Setup:
    pip install chromadb

Usage:
    python src/retrieval/vector_store.py
"""

import sys
from pathlib import Path

import chromadb

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.embedding.image_encoder import ImageEncoder
from src.embedding.text_encoder import TextEncoder
from src.ingestion.loader import load_all_reports

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_store"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.text_collection = self.client.get_or_create_collection("report_text")
        self.image_collection = self.client.get_or_create_collection("xray_images")

    def add_report_text(self, uid: str, text: str, embedding, extra_metadata: dict = None):
        metadata = {"uid": uid, "text_preview": text[:200]}
        if extra_metadata:
            metadata.update(extra_metadata)

        self.text_collection.add(
            ids=[f"report-{uid}"],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[metadata],
        )

    def add_image(self, uid: str, image_filename: str, embedding):
        self.image_collection.add(
            ids=[f"image-{image_filename}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"uid": uid, "filename": image_filename}],
        )

    def query_text(self, query_embedding, top_k: int = 5):
        return self.text_collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

    def query_image(self, query_embedding, top_k: int = 5):
        return self.image_collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )

    def counts(self):
        return {
            "text_entries": self.text_collection.count(),
            "image_entries": self.image_collection.count(),
        }


def populate(limit: int = 5):
    """Encodes and stores embeddings for a small batch of reports (for testing)."""
    reports = load_all_reports(limit=limit)
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    store = VectorStore()

    for report in reports:
        text_vec = text_encoder.encode(report.full_text())
        store.add_report_text(report.uid, report.full_text(), text_vec)

        for image_path in report.image_paths:
            img_vec = image_encoder.encode(image_path)
            store.add_image(report.uid, image_path.name, img_vec)

        print(f"Stored report {report.uid} ({len(report.image_paths)} images)")

    print("\nCounts:", store.counts())
    return store


if __name__ == "__main__":
    store = populate(limit=5)

    # quick sanity check: search using one of the reports' own text as the query
    from src.embedding.text_encoder import TextEncoder

    encoder = TextEncoder()
    query = "normal chest, no acute findings"
    query_vec = encoder.encode(query)

    results = store.query_text(query_vec, top_k=3)
    print(f"\nTop matches for query: '{query}'")
    for uid, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"  - {uid} (distance={dist:.4f}): {doc[:100]}...")