"""Key-free tests for chunking and retrieval ranking.

The stub embedder maps each text to a tiny keyword vector, so the
ranking behaviour can be proven without a Gemini API key. Run:

    python test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.chunker import chunk_text
from rag.store import VectorStore, cosine_similarity

KEYWORDS = ["gym", "rag", "python"]


def stub_embed(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(k)) for k in KEYWORDS]


def test_chunking() -> None:
    text = " ".join(f"word{i}" for i in range(700))
    chunks = chunk_text(text, source="test.txt")
    assert len(chunks) == 3, f"expected 3 chunks, got {len(chunks)}"
    # Overlap: the second chunk must repeat the tail of the first.
    first_words = set(chunks[0]["text"].split())
    second_words = set(chunks[1]["text"].split())
    assert first_words & second_words, "chunks should overlap"
    print("chunking: OK (3 chunks, overlap present)")


def test_retrieval_ranking() -> None:
    store = VectorStore("/tmp/rag_test_store.json")
    store.chunks = []
    docs = [
        ("gym doc", "gym members attendance subscription gym"),
        ("rag doc", "rag retrieval embeddings cosine similarity rag"),
        ("python doc", "python functions classes python"),
    ]
    chunks = [{"text": t, "source": s, "chunk_index": 0} for s, t in docs]
    store.add(chunks, [stub_embed(c["text"]) for c in chunks])
    results = store.search(stub_embed("what is rag retrieval"), top_k=1)
    assert results[0]["source"] == "rag doc", f"wrong top hit: {results[0]}"
    assert results[0]["score"] > 0, "top score should be positive"
    print(f"retrieval: OK (top hit 'rag doc', score {results[0]['score']})")


def test_cosine_edge_cases() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    print("cosine edge cases: OK")


if __name__ == "__main__":
    test_chunking()
    test_retrieval_ranking()
    test_cosine_edge_cases()
    print("All tests passed.")
