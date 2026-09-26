"""
FastAPI backend exposing the multimodal RAG (knowledge-graph powered)
pipeline as a simple HTTP API for the Streamlit frontend to call.

Run with:
    uvicorn backend.main:app --reload --port 8000

(Run this from the project root, medirag-kg/, not from inside backend/)
"""

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.llm_generator import SYSTEM_PROMPT, get_backend
from src.retrieval.hybrid_retriever import HybridRetriever

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