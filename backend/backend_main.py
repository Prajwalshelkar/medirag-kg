"""
FastAPI backend exposing the multimodal RAG (knowledge-graph powered)
pipeline as a simple HTTP API for the Streamlit frontend to call.

Run with:
    uvicorn backend.backend_main:app --reload --port 8000

(Run this from the project root, medirag-kg/, not from inside backend/)
"""

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embedding.image_encoder import ImageEncoder
from src.embedding.text_encoder import TextEncoder
from src.generation.llm_generator import SYSTEM_PROMPT, get_backend
from src.ingestion.loader import Report
from src.knowledge_graph.graph_builder import GraphBuilder
from src.knowledge_graph.ner_extractor import NERExtractor
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RECENT_LOG_PATH = UPLOAD_DIR / "recent_log.json"

# encoders/extractor are slow to load -- create them once, reuse across requests
_image_encoder = None
_text_encoder = None
_ner_extractor = None


def _get_shared_tools():
    global _image_encoder, _text_encoder, _ner_extractor
    if _image_encoder is None:
        _image_encoder = ImageEncoder()
    if _text_encoder is None:
        _text_encoder = TextEncoder()
    if _ner_extractor is None:
        _ner_extractor = NERExtractor()
    return _image_encoder, _text_encoder, _ner_extractor


def _append_to_recent_log(entry: dict):
    log = []
    if RECENT_LOG_PATH.exists():
        log = json.loads(RECENT_LOG_PATH.read_text())
    log.insert(0, entry)  # newest first
    log = log[:20]  # keep only the 20 most recent
    RECENT_LOG_PATH.write_text(json.dumps(log, indent=2))

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="MediRAG-KG API")

# allows the Streamlit frontend (running on a different port) to also call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serves the custom HTML/CSS/JS frontend directly from this same server
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class RetrievedReport(BaseModel):
    uid: str
    text_preview: str
    distance: float
    present_findings: List[str]
    negated_findings: List[str]


class QueryResponse(BaseModel):
    answer: str
    retrieved_reports: List[RetrievedReport]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    retriever = HybridRetriever()
    try:
        retrieved = retriever.retrieve(request.question, top_k=request.top_k)
        context_block = retriever.build_context_block(retrieved)
    finally:
        retriever.close()

    prompt = f"""Context from retrieved radiology reports:

{context_block}

Question: {request.question}

Answer based only on the context above."""

    backend = get_backend()
    answer = backend.generate(prompt)

    return QueryResponse(answer=answer, retrieved_reports=retrieved)


class UploadResponse(BaseModel):
    uid: str
    conclusion: str
    present_findings: List[str]
    negated_findings: List[str]


@app.post("/upload", response_model=UploadResponse)
async def upload_report(
    image: UploadFile = File(...),
    report_text: str = Form(...),
):
    """
    Accepts a new X-ray image + its report text, runs it through the same
    NER/negation extraction and adds it to both the knowledge graph and
    vector store (so future queries can retrieve it too), then generates
    a grounded conclusion for this specific report.
    """
    uid = f"uploaded-{uuid.uuid4().hex[:8]}"

    image_path = UPLOAD_DIR / f"{uid}_{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    image_encoder, text_encoder, ner_extractor = _get_shared_tools()

    entities = ner_extractor.extract_with_negation(report_text)
    present_findings = [e["entity"] for e in entities if not e["negated"]]
    negated_findings = [e["entity"] for e in entities if e["negated"]]

    # add to the knowledge graph so it's queryable alongside the original dataset
    report_obj = Report(uid=uid, findings=report_text, image_paths=[image_path])
    graph_builder = GraphBuilder()
    try:
        graph_builder.add_report(report_obj, entities)
    finally:
        graph_builder.close()

    # add to the vector store too
    store = VectorStore()
    text_vec = text_encoder.encode(report_text)
    store.add_report_text(uid, report_text, text_vec)
    img_vec = image_encoder.encode(image_path)
    store.add_image(uid, image_path.name, img_vec)

    # generate a grounded conclusion for this specific uploaded report
    prompt = f"""A new chest X-ray report was just uploaded. Summarize it into \
a short, clear conclusion, using only what is stated below. Clearly \
separate confirmed findings from findings that were explicitly ruled out.

Report text:
{report_text}

Conclusion:"""
    llm_backend = get_backend()
    conclusion = llm_backend.generate(prompt)

    _append_to_recent_log(
        {
            "uid": uid,
            "filename": image.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "present_findings": present_findings,
            "negated_findings": negated_findings,
        }
    )

    return UploadResponse(
        uid=uid,
        conclusion=conclusion,
        present_findings=present_findings,
        negated_findings=negated_findings,
    )


@app.get("/recent")
def get_recent_reports():
    """Returns the most recently uploaded/studied reports (newest first)."""
    if not RECENT_LOG_PATH.exists():
        return {"recent": []}
    return {"recent": json.loads(RECENT_LOG_PATH.read_text())}