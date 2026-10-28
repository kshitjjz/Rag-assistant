#!/usr/bin/env python3
"""Command line for the RAG assistant.

Usage:
    python main.py ingest docs/          # chunk + embed docs/ into store.json
    python main.py ask "your question"   # retrieve + answer from the store
    python main.py stats                 # show how many chunks are stored
"""

import argparse
import sys

from rag.pipeline import ask, ingest_folder
from rag.store import VectorStore

DEFAULT_STORE = "store.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAG assistant (Python + Gemini)")
    parser.add_argument("--store", default=DEFAULT_STORE,
                        help="path to the JSON store file")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest a folder of .txt/.md files")
    ingest.add_argument("folder", help="folder containing documents")

    ask_cmd = sub.add_parser("ask", help="ask a question over the store")
    ask_cmd.add_argument("question", help="the question to answer")
    ask_cmd.add_argument("--top-k", type=int, default=5,
                         help="how many chunks to retrieve")

    sub.add_parser("stats", help="show store contents summary")

    args = parser.parse_args(argv)
    store = VectorStore(args.store)

    try:
        if args.command == "ingest":
            count = ingest_folder(args.folder, store)
            print(f"Ingested {count} chunks into {args.store} "
                  f"({len(store)} total).")
        elif args.command == "ask":
            result = ask(args.question, store, top_k=args.top_k)
            print(result["answer"])
            print("\nSources:")
            for src in result["sources"]:
                print(f"  - {src['source']}#chunk{src['chunk_index']} "
                      f"(score {src['score']})")
        elif args.command == "stats":
            print(f"{len(store)} chunks stored in {args.store}.")
            sources = sorted({c["source"] for c in store.chunks})
            for source in sources:
                n = sum(1 for c in store.chunks if c["source"] == source)
                print(f"  - {source}: {n} chunks")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
