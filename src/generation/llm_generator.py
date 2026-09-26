"""
Generates grounded answers using the hybrid retriever's context block.

Supports two interchangeable backends, switchable via .env:
    LLM_BACKEND=groq    -- fast, free, hosted (needs GROQ_API_KEY)
    LLM_BACKEND=ollama  -- fully local, slower, no internet needed

Why both: Groq is fast and free but rate-limited (fine for dev/demo use);
Ollama has no rate limit at all since it runs on your own machine, useful
as a fallback if you hit Groq's daily cap or have no internet.

Setup:
    pip install groq          (for the Groq backend)
    ollama pull llama3.2      (for the Ollama backend, already done)

    In .env:
        LLM_BACKEND=groq
        GROQ_API_KEY=your-key-here

Usage:
    python src/generation/llm_generator.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are a radiology assistant helping summarize chest X-ray \
findings based on retrieved report context. Only state findings that are \
explicitly present in the provided context. Clearly distinguish confirmed \
findings from findings that were explicitly ruled out. If the context is \
insufficient to answer, say so rather than guessing. This is for an \
educational/portfolio project, not real clinical use."""


class GroqBackend:
    def __init__(self):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = "qwen/qwen3.8-27b"

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # low temperature: we want grounded, not creative
        )
        return response.choices[0].message.content


class OllamaBackend:
    def __init__(self, model: str = "llama3.2"):
        import ollama

        self.ollama = ollama
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]


def get_backend():
    backend_name = os.getenv("LLM_BACKEND", "groq").lower()
    if backend_name == "groq":
        return GroqBackend()
    elif backend_name == "ollama":
        return OllamaBackend()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend_name}. Use 'groq' or 'ollama'.")


def answer_query(query: str, top_k: int = 3) -> str:
    """
    Full pipeline: retrieve context (vector + graph) -> build prompt ->
    generate a grounded answer.
    """
    retriever = HybridRetriever()
    try:
        retrieved = retriever.retrieve(query, top_k=top_k)
        context_block = retriever.build_context_block(retrieved)
    finally:
        retriever.close()

    prompt = f"""Context from retrieved radiology reports:

{context_block}

Question: {query}

Answer based only on the context above."""

    backend = get_backend()
    return backend.generate(prompt)


if __name__ == "__main__":
    query = "Is there any evidence of pneumothorax in these reports?"
    print(f"Query: {query}\n")
    answer = answer_query(query)
    print("Answer:")
    print(answer)   