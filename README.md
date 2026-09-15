# medirag-kg

Multimodal RAG (knowledge-graph powered) system for healthcare — X-ray and clinical report analysis.

## Architecture

1. **Data ingestion** — X-ray images + paired clinical reports
2. **Embedding + KG construction** — image encoder, text NER, graph builder
3. **Hybrid retrieval** — vector store + knowledge graph traversal
4. **LLM generation** — grounded answer generation from retrieved context
5. **Web interface** — FastAPI backend + frontend UI

## Structure

```
medirag-kg/
├── data/                    # raw and processed datasets (gitignored)
├── src/
│   ├── ingestion/           # dataset loaders, preprocessing
│   ├── embedding/           # image + text encoders
│   ├── knowledge_graph/     # entity/relation extraction, graph builder
│   ├── retrieval/           # hybrid vector + graph retrieval
│   └── generation/          # LLM prompting and grounding
├── backend/                 # FastAPI app
├── frontend/                # UI (Streamlit or React — TBD)
├── notebooks/                # exploration notebooks
└── docs/                    # project tracker and design notes
```

## Status

See `docs/project-tracker.md` for current progress and open decisions.
