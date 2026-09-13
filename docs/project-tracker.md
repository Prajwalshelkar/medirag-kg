# Multimodal RAG (Knowledge Graph Powered) for Healthcare — Project Tracker

**Goal:** X-ray + clinical report analysis system using multimodal RAG, augmented by a knowledge graph, with a full-stack web interface.

**Why this project:** Rejected as college major project (application issue, not idea). Building independently for resume/placement value and to learn multimodal RAG, KG-augmented retrieval, and full-stack + LLM integration.

**Constraint to design around:** Free-tier usage — prefer local/open-source models over paid APIs wherever possible.

---

## Architecture (5 stages)

1. **Data ingestion** — X-ray images + paired clinical reports
2. **Embedding + KG construction** — image encoder, text NER, graph builder
3. **Hybrid retrieval** — vector store + knowledge graph traversal
4. **LLM generation** — grounded answer generation from retrieved context
5. **Web interface** — FastAPI backend + frontend UI

---

## Tech stack (proposed, free-tier friendly)

| Component | Choice | Notes |
|---|---|---|
| Dataset | MIMIC-CXR or IU X-ray | Free, paired X-ray + report, benchmarkable |
| Image encoder | CheXzero / BiomedCLIP | Run locally via Hugging Face |
| Text NER | scispaCy / BioClinicalBERT | Extract findings, anatomy, diagnoses |
| Knowledge graph | Neo4j | Free/local install, Cypher queries |
| Vector store | FAISS or Chroma | Free, local |
| LLM | Local (Ollama: Llama/Mistral) for dev; hosted API only for final demo | Saves tokens/credits |
| Backend | FastAPI | Async, lightweight |
| Frontend | Streamlit (fast) or React (more resume weight) | Decision pending |

---

## Status

- [x] Architecture sketched (5-stage pipeline)
- [x] Tech stack proposed
- [ ] Dataset finalized and downloaded
- [ ] Image encoder selected and tested
- [ ] NER / entity extraction pipeline built
- [ ] Knowledge graph schema designed
- [ ] Vector store set up
- [ ] Hybrid retrieval logic built
- [ ] LLM prompt/grounding strategy finalized
- [ ] Backend API scaffolded
- [ ] Frontend decision made + built
- [ ] End-to-end integration test
- [ ] Demo polish for resume/placement use

---

## Open decisions

- Frontend: Streamlit vs React — not yet decided
- Which local LLM (size/quantization) balances quality vs your hardware limits — not yet decided

---

## Log

- **Session 1:** Defined project scope, confirmed resume/placement value, decided to build fully within this chat rather than splitting across tools (Antigravity free tier too restrictive for sustained work). Sketched 5-stage architecture and proposed initial tech stack.
