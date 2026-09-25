"""
Combines two retrieval signals into one context for the LLM:

1. VECTOR SEARCH (ChromaDB) -- finds reports whose TEXT is semantically
   similar to the query. Good at "find reports that talk about similar
   things," but doesn't understand structured relationships.

2. GRAPH TRAVERSAL (Neo4j) -- for each report the vector search finds,
   pulls its exact structured findings (present AND negated) from the
   knowledge graph. This grounds the LLM in precise, verified facts
   rather than just "similar-sounding" text, which matters a lot for
   healthcare where "no consolidation" and "consolidation present" are
   opposite meanings that plain text similarity can blur.

The two combined = hybrid retrieval: the "what does this generally sound
like" of vector search, backed by the "what exactly was found or ruled
out" precision of the graph.

Setup:
    Requires vector_store.py to have been populated first (run it once),
    and requires the same .env Neo4j credentials as graph_builder.py.

Usage:
    python src/retrieval/hybrid_retriever.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.embedding.text_encoder import TextEncoder
from src.retrieval.vector_store import VectorStore

load_dotenv()


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.text_encoder = TextEncoder()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _get_graph_findings(self, uid: str) -> dict:
        """Pulls exact present/negated findings for a report from Neo4j."""
        query = """
        MATCH (r:Report {uid: $uid})-[rel:HAS_FINDING|NEGATIVE_FOR]->(f:Finding)
        RETURN type(rel) AS relation, f.name AS finding
        """
        with self.driver.session() as session:
            result = session.run(query, uid=uid)
            present, negated = [], []
            for record in result:
                if record["relation"] == "HAS_FINDING":
                    present.append(record["finding"])
                else:
                    negated.append(record["finding"])
        return {"present_findings": present, "negated_findings": negated}

    def retrieve(self, query_text: str, top_k: int = 3) -> list:
        """
        Returns a list of dicts, one per matched report:
        {uid, text_preview, distance, present_findings, negated_findings}
        """
        query_vec = self.text_encoder.encode(query_text)
        vector_results = self.vector_store.query_text(query_vec, top_k=top_k)

        combined = []
        for uid_key, doc, dist in zip(
            vector_results["ids"][0],
            vector_results["documents"][0],
            vector_results["distances"][0],
        ):
            uid = uid_key.replace("report-", "")
            graph_data = self._get_graph_findings(uid)

            combined.append(
                {
                    "uid": uid,
                    "text_preview": doc[:300],
                    "distance": dist,
                    **graph_data,
                }
            )

        return combined

    def build_context_block(self, retrieved: list) -> str:
        """
        Formats retrieved results into a text block ready to paste into an
        LLM prompt for grounded generation.
        """
        lines = []
        for item in retrieved:
            lines.append(f"--- Report {item['uid']} (similarity distance: {item['distance']:.2f}) ---")
            lines.append(f"Text: {item['text_preview']}")
            if item["present_findings"]:
                lines.append(f"Confirmed findings: {', '.join(item['present_findings'])}")
            if item["negated_findings"]:
                lines.append(f"Explicitly ruled out: {', '.join(item['negated_findings'])}")
            lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    retriever = HybridRetriever()

    try:
        query = "normal chest, no acute findings"
        results = retriever.retrieve(query, top_k=3)

        print(f"Query: '{query}'\n")
        context_block = retriever.build_context_block(results)
        print(context_block)
    finally:
        retriever.close()