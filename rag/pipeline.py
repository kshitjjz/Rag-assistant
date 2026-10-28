"""Orchestration: ingest a folder of documents, answer a question."""

import glob
import os

from .chunker import chunk_text, load_text_file
from .embedder import embed_query, embed_texts, generate_answer
from .store import VectorStore

SUPPORTED_EXTENSIONS = (".txt", ".md")


def ingest_folder(folder: str, store: VectorStore) -> int:
    """Chunk and embed every supported file in folder. Returns chunk count."""
    paths = sorted(
        p for pattern in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(folder, f"*{pattern}"))
    )
    if not paths:
        raise RuntimeError(f"No .txt or .md files found in {folder}")
    all_chunks = []
    for path in paths:
        text = load_text_file(path)
        all_chunks.extend(chunk_text(text, source=os.path.basename(path)))
    embeddings = embed_texts([c["text"] for c in all_chunks])
    store.add(all_chunks, embeddings)
    store.save()
    return len(all_chunks)


def ask(question: str, store: VectorStore, top_k: int = 5) -> dict:
    """Retrieve, then generate a grounded answer. Returns answer + sources."""
    if len(store) == 0:
        raise RuntimeError("The store is empty. Run `ingest` first.")
    query_embedding = embed_query(question)
    retrieved = store.search(query_embedding, top_k=top_k)
    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": retrieved}
