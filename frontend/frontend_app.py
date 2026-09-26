"""
Streamlit frontend for MediRAG-KG. Calls the FastAPI backend, which must
be running separately.

Run with (from the project root, medirag-kg/):
    streamlit run frontend/app.py

Make sure the backend is already running in another terminal:
    uvicorn backend.main:app --reload --port 8000
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MediRAG-KG", page_icon="🩻", layout="centered")

st.title("🩻 MediRAG-KG")
st.caption(
    "Multimodal RAG (knowledge-graph powered) assistant for chest X-ray "
    "report analysis — portfolio project, not for real clinical use."
)

question = st.text_area(
    "Ask a question about the stored chest X-ray reports:",
    placeholder="e.g. Is there any evidence of pneumothorax?",
    height=100,
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving context and generating a grounded answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query", json={"question": question, "top_k": 3}
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Retrieved evidence")
            for report in data["retrieved_reports"]:
                with st.expander(
                    f"Report {report['uid']} — similarity distance {report['distance']:.2f}"
                ):
                    st.write(report["text_preview"])
                    if report["present_findings"]:
                        st.markdown(
                            f"**Confirmed findings:** {', '.join(report['present_findings'])}"
                        )
                    if report["negated_findings"]:
                        st.markdown(
                            f"**Explicitly ruled out:** {', '.join(report['negated_findings'])}"
                        )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend API. Make sure it's running in "
                "another terminal: uvicorn backend.main:app --reload --port 8000"
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "Pipeline: ChromaDB vector search + Neo4j knowledge graph traversal "
    "-> grounded LLM generation (Groq/Ollama)."
)