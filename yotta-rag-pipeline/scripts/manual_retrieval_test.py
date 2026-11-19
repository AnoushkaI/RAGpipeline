"""
Manual Retrieval Test Script for Yotta Workstream A RAG Pipeline.

Executes a live interactive comparison between Dense Vector Retrieval
and BM25 Lexical Keyword Retrieval using:
- The local 20-passage corpus (sample_passages.jsonl)
- Fixed-size chunking (chunk_size=300, chunk_overlap=50)
- Real all-MiniLM-L6-v2 embedding model on CPU
- Top-5 ranked candidate chunks for each retriever
- Terminal input() interaction with graceful empty-input handling
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_config
from rag.chunking import chunk_corpus
from rag.indexing import build_indices
from rag.ingestion import load_corpus
from rag.providers import get_embedding_provider
from rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult


def print_results_block(title: str, results: list[RetrievalResult], strategy_desc: str) -> None:
    """Prints a formatted block of retrieved candidates."""
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print(f" {strategy_desc}")
    print("=" * 80)

    if not results:
        print("  [!] No matching chunks found.")
        print("=" * 80)
        return

    for rank, res in enumerate(results, start=1):
        score_label = f"Cosine: {res.score:.4f}" if res.strategy == "dense" else f"BM25 Score: {res.score:.4f}"
        print(f"\n[Rank {rank}] | {score_label} | Chunk ID: {res.chunk_id} | Parent Doc ID: {res.doc_id}")
        print("-" * 80)
        print(f"{res.text.strip()}")
        print("-" * 80)

    print("=" * 80 + "\n")


def main() -> None:
    print("=" * 80)
    print("       YOTTA RAG PIPELINE — PHASE 3 MANUAL RETRIEVAL TEST")
    print("=" * 80)

    # 1. Load Configuration
    config = load_config()
    chunk_size = config.rag.chunk_size
    chunk_overlap = config.rag.chunk_overlap
    top_k = 5

    print(f"[1/4] Loading local passage corpus...")
    passages = load_corpus(use_hf=False)
    print(f"      Loaded {len(passages)} passages.")

    print(f"[2/4] Generating fixed-size chunks (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_corpus(
        passages,
        strategy="fixed",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"      Generated {len(chunks)} chunks across {len(passages)} documents.")

    print(f"[3/4] Initializing local embedding provider: {config.embedding.provider} ({config.embedding.model})...")
    embedding_provider = get_embedding_provider(config.embedding)

    print(f"[4/4] Building Dense (NumPy) and BM25 (BM25Okapi) indexes over {len(chunks)} chunks...")
    dense_index, bm25_index = build_indices(chunks, embedding_provider=embedding_provider)
    print("      Both indexes built successfully!")

    dense_retriever = DenseRetriever(index=dense_index, embedding_provider=embedding_provider)
    bm25_retriever = BM25Retriever(index=bm25_index)

    # Allow CLI argument or interactive terminal input
    if len(sys.argv) > 1 and sys.argv[1].strip():
        user_question = " ".join(sys.argv[1:]).strip()
        print(f"\n[Input Query from CLI]: \"{user_question}\"")
    else:
        try:
            print("\n" + "-" * 80)
            user_question = input("Enter your search question (or press Enter to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not user_question:
        print("\n[!] Empty query received. No retrieval performed. Exiting gracefully.")
        return

    print(f"\nExecuting retrieval for: \"{user_question}\" (Top-{top_k})...\n")

    # 1. Run Dense Retrieval
    dense_results = dense_retriever.retrieve(query=user_question, top_k=top_k)
    print_results_block(
        title="Dense Vector Retrieval Results",
        results=dense_results,
        strategy_desc="Strategy: Dense Cosine Similarity (all-MiniLM-L6-v2 on CPU)",
    )

    # 2. Run BM25 Retrieval
    bm25_results = bm25_retriever.retrieve(query=user_question, top_k=top_k)
    print_results_block(
        title="BM25 Lexical Keyword Retrieval Results",
        results=bm25_results,
        strategy_desc="Strategy: BM25Okapi (k1=1.5, b=0.75, alphanumeric tokens)",
    )


if __name__ == "__main__":
    main()
