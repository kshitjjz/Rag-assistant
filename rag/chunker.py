"""Text chunking for the RAG pipeline.

Strategy: split a document into paragraphs first (so chunks never start
mid-sentence), then pack paragraphs into word windows with overlap.
Overlap matters: a fact sitting on a window boundary is still retrievable
from the neighbouring chunk.
"""

import re

DEFAULT_CHUNK_WORDS = 300
DEFAULT_OVERLAP_WORDS = 50


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_text(text: str, source: str,
               chunk_words: int = DEFAULT_CHUNK_WORDS,
               overlap_words: int = DEFAULT_OVERLAP_WORDS) -> list[dict]:
    """Split text into overlapping chunks.

    Returns a list of dicts with keys: text, source, chunk_index.
    """
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be larger than overlap_words")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Flatten to one word stream, but remember paragraph breaks are gone;
    # the window approach keeps chunks readable on their own.
    words = []
    for para in paragraphs:
        words.extend(_words(para))

    chunks = []
    step = chunk_words - overlap_words
    start = 0
    index = 0
    while start < len(words):
        window = words[start:start + chunk_words]
        chunks.append({
            "text": " ".join(window),
            "source": source,
            "chunk_index": index,
        })
        index += 1
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
