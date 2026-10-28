"""A tiny JSON-backed vector store.

No database, no numpy: embeddings are plain lists of floats, similarity
is cosine similarity computed in pure Python, and everything persists to
one JSON file. That keeps the project runnable anywhere with just the
Gemini client installed, and keeps the retrieval logic easy to explain
in an interview.
"""

import json
import math
import os


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Chunks + their embeddings, persisted as JSON."""

    def __init__(self, path: str):
        self.path = path
        self.chunks: list[dict] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for chunk, embedding in zip(chunks, embeddings):
            record = dict(chunk)
            record["embedding"] = embedding
            self.chunks.append(record)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return the top_k chunks ranked by cosine similarity."""
        scored = [
            (cosine_similarity(query_embedding, c["embedding"]), c)
            for c in self.chunks
            if c.get("embedding")
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            result = {k: v for k, v in chunk.items() if k != "embedding"}
            result["score"] = round(score, 4)
            results.append(result)
        return results

    def __len__(self) -> int:
        return len(self.chunks)
