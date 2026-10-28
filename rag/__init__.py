"""rag: a small retrieval-augmented generation assistant."""

from .chunker import chunk_text
from .store import VectorStore

__all__ = ["chunk_text", "VectorStore"]
