# RAG Assistant (Python + Gemini)

A small, dependency-light retrieval-augmented generation assistant. It
ingests your own `.txt` / `.md` documents, embeds them with Gemini,
retrieves the most relevant chunks for a question, and answers **only
from those chunks**, with source citations. If the documents don't
contain the answer, it says so instead of guessing.

## How it works

```
docs/*.txt, *.md
      │  chunk_text() — overlapping word windows (300 words, 50 overlap)
      ▼
embed_texts() — Gemini text-embedding-004
      │  stored as JSON (store.json) with cosine similarity search
      ▼
ask "question"
      │  embed query → top-k chunks → grounded prompt → gemini-2.5-flash
      ▼
answer + cited sources
```

Why these choices:

- **Overlapping chunks** so a fact on a window boundary is still
  retrievable from the neighbouring chunk.
- **Cosine similarity in pure Python** — no database, no numpy. The
  retrieval logic is short enough to explain line by line in an
  interview.
- **Grounding enforced in the prompt** — the model must cite a source
  tag for every claim, and must say "I don't know based on the
  provided documents" when the context is silent. That rule is what
  makes a RAG system trustworthy.
- **API key only via environment variable** — never committed, never
  stored in the vector store.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

Model names are overridable without code edits:

```bash
export GEMINI_EMBED_MODEL=text-embedding-004
export GEMINI_MODEL=gemini-2.5-flash
```

## Usage

```bash
# Ingest the sample docs (or point at your own folder)
python main.py ingest docs/

# Ask a question over the ingested documents
python main.py ask "What is retrieval-augmented generation?"

# Inspect the store
python main.py stats
```

## Tests (no API key needed)

`test_retrieval.py` verifies chunking and retrieval ranking with a
stub embedder, so the core logic can be proven without calling Gemini:

```bash
python test_retrieval.py
```

## Limitations (honest ones)

- `.txt` and `.md` only — no PDF parsing yet.
- The JSON store is fine for hundreds of documents, not millions; a
  real deployment would swap it for a vector database.
- Retrieval quality depends on the embedding model; a bad embedding
  still retrieves *something*, which is why the grounding rule matters.
- Answers are only as current as the documents you ingest.

## Project structure

```
main.py               CLI: ingest / ask / stats
rag/
  chunker.py          overlapping word-window chunking
  store.py            JSON vector store + cosine similarity
  embedder.py         Gemini embeddings + grounded generation
  pipeline.py         ingest_folder() / ask() orchestration
docs/                 sample documents to try it on
test_retrieval.py     key-free tests for chunking + ranking
```
