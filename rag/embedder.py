"""Gemini wrappers: embeddings and grounded answer generation.

The API key is read only from the GEMINI_API_KEY environment variable.
It is never stored in the repo, in the store file, or in any log.
Model names are overridable with GEMINI_EMBED_MODEL / GEMINI_MODEL so
the project survives model renames without code edits.
"""

import os

EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004")
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        ) from exc
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it before running, e.g.:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the Gemini embedding model."""
    client = _client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [list(e.values) for e in result.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Answer strictly from the retrieved chunks, citing sources.

    Grounding rule: if the chunks do not contain the answer, the model
    must say so instead of guessing. This is the behaviour that makes a
    RAG system trustworthy, and it is enforced in the prompt, not hoped for.
    """
    client = _client()
    context = "\n\n".join(
        f"[{c['source']}#chunk{c['chunk_index']}] {c['text']}"
        for c in context_chunks
    )
    prompt = (
        "You answer questions using ONLY the context below. "
        "If the context does not contain the answer, say you don't know "
        "based on the provided documents. Cite every factual claim with "
        "the bracketed source tag it came from.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text.strip()
