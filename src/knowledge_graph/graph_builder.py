"""
Builds a knowledge graph in Neo4j from parsed radiology reports.

Schema:
    (:Report {uid})
    (:Finding {name})
    (:Image {filename})

    (Report)-[:HAS_FINDING]->(Finding)       -- finding present in report
    (Report)-[:NEGATIVE_FOR]->(Finding)      -- finding explicitly ruled out
    (Report)-[:HAS_IMAGE]->(Image)           -- image belongs to this report

Setup:
    pip install neo4j python-dotenv
    Create a .env file (see project README) with:
        NEO4J_URI=...
        NEO4J_USERNAME=...
        NEO4J_PASSWORD=...

Usage:
    python src/knowledge_graph/graph_builder.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.loader import load_all_reports
from src.knowledge_graph.ner_extractor import NERExtractor

load_dotenv()


class GraphBuilder:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not all([uri, user, password]):
            raise EnvironmentError(
                "Missing Neo4j credentials. Make sure .env has NEO4J_URI, "
                "NEO4J_USERNAME, and NEO4J_PASSWORD set."
            )

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes all nodes/relationships. Useful when re-running during development."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def add_report(self, report, entities_with_negation: list):
        """
        report: a Report object from src/ingestion/loader.py
        entities_with_negation: output of NERExtractor.extract_with_negation()
        """
        with self.driver.session() as session:
            session.execute_write(self._create_report_node, report.uid)

            for image_path in report.image_paths:
                session.execute_write(
                    self._link_image, report.uid, image_path.name
                )

            for item in entities_with_negation:
                session.execute_write(
                    self._link_finding, report.uid, item["entity"], item["negated"]
                )

    @staticmethod
    def _create_report_node(tx, uid):
        tx.run("MERGE (r:Report {uid: $uid})", uid=uid)

    @staticmethod
    def _link_image(tx, uid, filename):
        tx.run(
            """
            MATCH (r:Report {uid: $uid})
            MERGE (i:Image {filename: $filename})
            MERGE (r)-[:HAS_IMAGE]->(i)
            """,
            uid=uid,
            filename=filename,
        )

    @staticmethod
    def _link_finding(tx, uid, finding_name, negated):
        relationship = "NEGATIVE_FOR" if negated else "HAS_FINDING"
        tx.run(
            f"""
            MATCH (r:Report {{uid: $uid}})
            MERGE (f:Finding {{name: $finding_name}})
            MERGE (r)-[:{relationship}]->(f)
            """,
            uid=uid,
            finding_name=finding_name.lower(),
        )


def build_graph(limit: int = 5):
    """Builds the graph for a small batch of reports (for testing)."""
    reports = load_all_reports(limit=limit)
    extractor = NERExtractor()
    builder = GraphBuilder()

    try:
        for report in reports:
            entities = extractor.extract_with_negation(report.full_text())
            builder.add_report(report, entities)
            print(f"Added report {report.uid} with {len(entities)} entities")
    finally:
        builder.close()

    print(f"\nDone. Built graph for {len(reports)} reports.")


if __name__ == "__main__":
    build_graph(limit=5)