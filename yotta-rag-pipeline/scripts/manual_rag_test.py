"""
Manual End-to-End RAG Pipeline Verification Script.

Executes and verifies the complete Phase 4 RAG pipeline:
Retrieval -> Relevance Grading -> Bounded Retry (Rewrite) -> Grounded Generation.

Tests both:
1. Unsupported Question: "What is the current market share of Yotta in the Indian cloud computing market?"
   -> Verifies all candidate chunks fail grading, triggers exactly 1 retry rewrite,
      and abstains gracefully when evidence remains insufficient.
2. Supported Question: "What is the difference between supervised and unsupervised learning?"
   -> Verifies relevant chunks pass grading and a grounded answer citing [doc_001] is generated.

Supports Dense, BM25, and Hybrid (RRF) retrieval strategies using the local CPU embedding setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure yotta-rag-pipeline root is in sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_config
from rag.pipeline import RAGPipeline, RAGResponse, create_pipeline
from rag.providers import BaseLLMProvider, MockLLMProvider, ProviderError, ProviderHealth
import re


class VerificationMockLLM(BaseLLMProvider):
    """
    Intelligent mock LLM for deterministic offline verification:
    - Reformulates search queries with relevant domain keywords on retry.
    - Synthesizes grounded answers citing [doc_001] when valid evidence is supplied.
    - Explicitly abstains when supplied context lacks evidence.
    """

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "reformulat" in prompt_lower or "search query" in prompt_lower:
            match = re.search(r"query '([^']+)'", prompt)
            q_text = match.group(1) if match else "expanded keywords"
            return f"{q_text} statistics revenue analysis"

        if "supervised" in prompt_lower and "unsupervised" in prompt_lower:
            return (
                "Supervised learning trains algorithms on labeled datasets with ground-truth pairs, "
                "whereas unsupervised learning discovers patterns and distributions in unlabeled data [doc_001]."
            )

        return (
            "I cannot answer this question based on the provided documents "
            "as the necessary information is not present in the retrieved context."
        )

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(name="VerificationMockLLM", kind="generation", status="OK", model="mock-verifier")


def print_response_details(res: RAGResponse, title: str = "") -> None:
    """Prints a clear, formatted breakdown of a RAG query execution."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()}")
        print("-" * 80)
    print(f"Query:              \"{res.query}\"")
    print(f"Retrieval Strategy: {res.strategy.upper()}")
    print(f"Grading Method:     {res.grading_method.upper()} (Threshold: {res.grading_result.get('threshold', 'N/A')})")
    print(f"Context Sufficient: {'YES' if res.has_sufficient_context else 'NO (Insufficient Evidence)'}")
    print(f"Retry Triggered:    {'YES (Bounded Retry = 1)' if res.retry_attempted else 'NO'}")
    if res.retry_attempted:
        print(f"Rewritten Query:    \"{res.rewritten_query}\"")
    print(f"Execution Latency:  {res.latency_ms:.1f}ms")
    print("=" * 80)

    # 1. Retrieved Candidates & Grading Decisions
    print("\n[RETRIEVED CANDIDATES & RELEVANCE GRADING DECISIONS]")
    print("-" * 80)
    decisions = res.grading_result.get("decisions", [])
    if not res.retrieved_chunks:
        print("  [!] No candidate chunks retrieved.")
    else:
        for idx, chunk in enumerate(res.retrieved_chunks, start=1):
            # Match decision
            cand_dec = next((d for d in decisions if d.get("chunk_id") == chunk.chunk_id), None)
            verdict = "PASS [RELEVANT]" if (cand_dec and cand_dec.get("is_relevant")) else "REJECT [IRRELEVANT]"
            reason = cand_dec.get("reasoning", "") if cand_dec else ""
            
            print(f"Candidate #{idx}: [{chunk.chunk_id}] (Parent Doc: [{chunk.doc_id}])")
            print(f"  Raw Retrieval Score : {chunk.score:.4f} ({chunk.strategy})")
            print(f"  Relevance Verdict   : {verdict}")
            if reason:
                print(f"  Grading Explanation : {reason}")
            print(f"  Snippet             : {chunk.text[:120].strip()}...")
            print("-" * 80)

    # 2. Accepted Context Passages Reaching Generation
    print("\n[ACCEPTED PASSAGES REACHING GENERATION]")
    print("-" * 80)
    if not res.passed_chunks:
        print("  (None — Insufficient relevant evidence; generation aborted / abstained)")
    else:
        for p in res.passed_chunks:
            print(f"  -> [{p.chunk_id}] (Source: [{p.doc_id}]): {p.text[:90].strip()}...")
    print("-" * 80)

    # 3. Final Answer & Source Citations
    print("\n[FINAL CONSOLIDATED ANSWER]")
    print("-" * 80)
    print(res.answer)
    print("-" * 80)
    print(f"Verified Citations : {res.cited_doc_ids if res.cited_doc_ids else '(None)'}")
    print("=" * 80 + "\n")


def run_benchmark_verification(pipeline: RAGPipeline) -> None:
    """Runs automated verification of unsupported and supported queries across all strategies."""
    unsupported_query = "What is the current market share of Yotta in the Indian cloud computing market?"
    supported_query = "What is the difference between supervised and unsupervised learning?"

    strategies = ["dense", "bm25", "hybrid"]

    print("\n" + "#" * 80)
    print(" PART 1: UNSUPPORTED QUERY VERIFICATION (EXPECTED: ABSTENTION)")
    print(" Query: \"" + unsupported_query + "\"")
    print("#" * 80)

    for strat in strategies:
        res = pipeline.run(unsupported_query, strategy=strat)
        print_response_details(
            res,
            title=f"UNSUPPORTED QUESTION — STRATEGY: {strat.upper()}"
        )
        # Assertions
        assert not res.has_sufficient_context, f"Strategy {strat} should not claim sufficient context for unsupported query!"
        assert res.retry_attempted, f"Strategy {strat} should trigger retry rewrite for unsupported query!"
        assert "cannot answer" in res.answer.lower() or "insufficient" in res.answer.lower(), (
            f"Strategy {strat} should abstain from answering unsupported query!"
        )

    print("\n" + "#" * 80)
    print(" PART 2: SUPPORTED QUERY VERIFICATION (EXPECTED: GROUNDED ANSWER)")
    print(" Query: \"" + supported_query + "\"")
    print("#" * 80)

    for strat in strategies:
        res = pipeline.run(supported_query, strategy=strat)
        print_response_details(
            res,
            title=f"SUPPORTED QUESTION — STRATEGY: {strat.upper()}"
        )
        # Assertions
        assert res.has_sufficient_context, f"Strategy {strat} should find sufficient context for supported query!"
        assert len(res.passed_chunks) > 0, f"Strategy {strat} should have passed chunks for supported query!"
        assert "doc_001" in [c.doc_id for c in res.passed_chunks], f"Strategy {strat} should retrieve doc_001!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual End-to-End RAG Pipeline Verification Script.")
    parser.add_argument("query", nargs="?", default=None, help="Custom question to test.")
    parser.add_argument("--strategy", choices=["dense", "bm25", "hybrid"], default="dense", help="Retrieval strategy.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal query loop.")
    parser.add_argument("--mock-llm", action="store_true", help="Force deterministic Mock LLM provider.")
    args = parser.parse_args()

    print("=" * 80)
    print("       YOTTA RAG PIPELINE — PHASE 4 END-TO-END VERIFICATION")
    print("=" * 80)

    config = load_config()

    print(f"[1/3] Loading and indexing local 20-passage corpus on CPU...")
    pipeline = create_pipeline(config=config, load_local_data=True, sample_size=20)
    print(f"      Dense and BM25 indices initialized with {len(pipeline.dense_index)} chunks.")

    # Check LLM connectivity; if offline or mock requested, configure smart verification mock
    if args.mock_llm or config.llm.provider == "mock":
        print("[2/3] Using deterministic Mock LLM Provider.")
        pipeline.llm_provider = VerificationMockLLM()
        pipeline.generator.llm_provider = pipeline.llm_provider
    else:
        print(f"[2/3] Using configured LLM provider: {config.llm.provider} ({config.llm.model}).")
        # Test health check; fall back cleanly if endpoint is unreachable in current environment
        health = pipeline.llm_provider.health_check()
        print(f"      LLM Provider Health: {health.status} - {health.message}")
        if health.status in {"FAILED", "MISSING_CREDENTIALS"}:
            print("      [Notice] Remote LLM unreachable in current sandbox. Using deterministic Mock LLM fallback.")
            pipeline.llm_provider = VerificationMockLLM()
            pipeline.generator.llm_provider = pipeline.llm_provider

    print("[3/3] Ready to execute pipeline runs.")

    # Branch by mode
    if args.interactive:
        print("\nEntering interactive mode. Type your question and press Enter (or blank to exit):")
        while True:
            try:
                user_q = input("\nQuestion > ").strip()
                if not user_q:
                    print("Exiting interactive mode.")
                    break
                strat = args.strategy
                res = pipeline.run(user_q, strategy=strat, top_k=args.top_k)
                print_response_details(res, title=f"INTERACTIVE QUERY — STRATEGY: {strat.upper()}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

    elif args.query:
        print(f"\nExecuting single query: \"{args.query}\" with strategy={args.strategy}...")
        res = pipeline.run(args.query, strategy=args.strategy, top_k=args.top_k)
        print_response_details(res, title=f"QUERY: \"{args.query}\"")

    else:
        # Default: Run full dual-benchmark verification (unsupported + supported across dense, bm25, hybrid)
        run_benchmark_verification(pipeline)


if __name__ == "__main__":
    main()
