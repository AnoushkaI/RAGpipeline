"""
Evaluation Module for Yotta Workstream A RAG Pipeline.

Implements rigorous, fair retrieval evaluation comparing Dense Vector Retrieval
and BM25 Lexical Retrieval over the exact same:
1. Document corpus & chunking strategy
2. Test queries and ground-truth relevance labels
3. Top-K cutoff levels (K = 1, 3, 5)

Ensures that retrieved chunk IDs are mapped to parent document IDs, and
prevents duplicate chunks from the same document from falsely inflating Recall@K.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# Ensure project root is in sys.path when executed directly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.chunking import Chunk
from rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Core Metric Computation
# ------------------------------------------------------------------------------

def calculate_recall_at_k(
    retrieved: Sequence[RetrievalResult | str],
    relevant_doc_ids: Sequence[str],
    k: int,
    chunk_to_doc_map: Optional[dict[str, str]] = None,
) -> float:
    """
    Computes Recall@K for a single query.
    
    Recall@K = (Number of relevant documents found in top-K) / (Total relevant documents)
    
    Critical Safeguards:
    1. Maps each retrieved chunk to its parent document ID.
    2. Takes the unique set of parent document IDs in the top-K to avoid counting
       multiple chunks from the same document more than once.
    3. If relevant_doc_ids is empty, returns 0.0.
    
    Args:
        retrieved: Sequence of RetrievalResult objects OR chunk ID strings.
        relevant_doc_ids: Ground-truth list of relevant parent document IDs.
        k: Cutoff rank (e.g. 1, 3, 5).
        chunk_to_doc_map: Optional mapping from chunk_id -> doc_id if retrieved
                          elements are raw chunk ID strings.
                          
    Returns:
        Float value between 0.0 and 1.0.
    """
    if k <= 0 or not retrieved or not relevant_doc_ids:
        return 0.0

    target_relevant = set(relevant_doc_ids)
    if not target_relevant:
        return 0.0

    # Slice top-K items
    top_k_items = retrieved[:k]

    # Map retrieved items to unique parent document IDs in top-k
    retrieved_doc_ids: set[str] = set()
    for item in top_k_items:
        if isinstance(item, RetrievalResult):
            retrieved_doc_ids.add(item.doc_id)
        elif isinstance(item, str):
            if chunk_to_doc_map and item in chunk_to_doc_map:
                retrieved_doc_ids.add(chunk_to_doc_map[item])
            else:
                # If no mapping, assume item itself is doc_id
                retrieved_doc_ids.add(item)
        elif hasattr(item, "doc_id"):
            retrieved_doc_ids.add(getattr(item, "doc_id"))

    # Count relevant documents found (unique intersection)
    hits = retrieved_doc_ids.intersection(target_relevant)
    return len(hits) / float(len(target_relevant))


# ------------------------------------------------------------------------------
# Evaluation Records and Reports
# ------------------------------------------------------------------------------

@dataclass
class QueryEvaluationRecord:
    """Detailed evaluation result for a single test query."""
    query_id: str
    query: str
    relevant_doc_ids: list[str]
    category: str
    dense_recall_at_k: dict[int, float]
    bm25_recall_at_k: dict[int, float]
    dense_retrieved_doc_ids: dict[int, list[str]]
    bm25_retrieved_doc_ids: dict[int, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report comparing Dense and BM25 retrievers."""
    num_queries: int
    k_values: list[int]
    dense_mean_recall: dict[int, float]
    bm25_mean_recall: dict[int, float]
    query_records: list[QueryEvaluationRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "dense_mean_recall": {str(k): round(v, 4) for k, v in self.dense_mean_recall.items()},
            "bm25_mean_recall": {str(k): round(v, 4) for k, v in self.bm25_mean_recall.items()},
            "query_records": [q.to_dict() for q in self.query_records],
        }

    def format_summary_table(self) -> str:
        """Renders an ASCII comparison table of Recall@K performance."""
        lines = [
            "==================================================================",
            "        RETRIEVAL EVALUATION REPORT: DENSE vs. BM25              ",
            "==================================================================",
            f"Evaluated Queries : {self.num_queries}",
            f"Evaluated K Levels: {self.k_values}",
            "------------------------------------------------------------------",
            f"{'Metric':<18} | {'Dense (Cosine)':<18} | {'BM25 (Okapi)':<18}",
            "------------------------------------------------------------------",
        ]
        for k in self.k_values:
            dense_r = self.dense_mean_recall.get(k, 0.0)
            bm25_r = self.bm25_mean_recall.get(k, 0.0)
            lines.append(f"Mean Recall@{k:<5} | {dense_r * 100:>15.2f}% | {bm25_r * 100:>15.2f}%")
        lines.append("==================================================================")
        return "\n".join(lines)

    def to_dataframe(self):
        """Converts query records to a pandas DataFrame for UI rendering."""
        import pandas as pd
        rows = []
        for q in self.query_records:
            gt_docs = ", ".join(q.relevant_doc_ids) if q.relevant_doc_ids else "None (Negative / Error Test)"
            row = {
                "Query ID": q.query_id,
                "Query": q.query,
                "Category": q.category,
                "Ground Truth Docs": gt_docs,
            }
            for k in self.k_values:
                row[f"Dense Recall@{k}"] = round(q.dense_recall_at_k.get(k, 0.0), 4)
                row[f"BM25 Recall@{k}"] = round(q.bm25_recall_at_k.get(k, 0.0), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(self, output_path: str | Path = "results/evaluation_results.csv") -> Path:
        """
        Exports query evaluation records to a structured CSV file.
        Creates parent directories automatically if missing.
        """
        import csv
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["query_id", "query", "category", "relevant_doc_ids"]
        for k in self.k_values:
            fieldnames.extend([f"dense_recall@{k}", f"bm25_recall@{k}"])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in self.query_records:
                row = {
                    "query_id": q.query_id,
                    "query": q.query,
                    "category": q.category,
                    "relevant_doc_ids": ";".join(q.relevant_doc_ids),
                }
                for k in self.k_values:
                    row[f"dense_recall@{k}"] = round(q.dense_recall_at_k.get(k, 0.0), 4)
                    row[f"bm25_recall@{k}"] = round(q.bm25_recall_at_k.get(k, 0.0), 4)
                writer.writerow(row)

        logger.info(f"Exported evaluation results to: {path}")
        return path


# ------------------------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------------------------

def load_eval_queries(path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Loads ground-truth evaluation queries from JSON file."""
    if path is None:
        target_path = Path(__file__).parent.parent / "data" / "eval_queries.json"
    else:
        target_path = Path(path)

    if not target_path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of queries in {target_path}, got {type(data)}")

    return data


# ------------------------------------------------------------------------------
# Evaluation Runner
# ------------------------------------------------------------------------------

def evaluate_retrievers(
    queries: list[dict[str, Any]],
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
    k_values: list[int] = [1, 3, 5],
) -> EvaluationReport:
    """
    Evaluates Dense and BM25 retrievers on the exact same set of queries and K levels.
    
    Args:
        queries: List of query dictionaries with 'query', 'relevant_doc_ids', etc.
        dense_retriever: Initialized DenseRetriever.
        bm25_retriever: Initialized BM25Retriever.
        k_values: List of K thresholds to evaluate (e.g. [1, 3, 5]).
        
    Returns:
        EvaluationReport containing aggregate and per-query Recall@K scores.
    """
    max_k = max(k_values)
    num_queries = len(queries)
    if num_queries == 0:
        return EvaluationReport(
            num_queries=0,
            k_values=k_values,
            dense_mean_recall={k: 0.0 for k in k_values},
            bm25_mean_recall={k: 0.0 for k in k_values},
        )

    dense_recall_sums = {k: 0.0 for k in k_values}
    bm25_recall_sums = {k: 0.0 for k in k_values}
    query_records: list[QueryEvaluationRecord] = []

    for q_item in queries:
        query_id = q_item.get("query_id", "unknown")
        query_text = q_item.get("query", "")
        relevant_doc_ids = q_item.get("relevant_doc_ids", [])
        category = q_item.get("category", "general")

        # Retrieve up to max_k candidates for both
        dense_results = dense_retriever.retrieve(query=query_text, top_k=max_k)
        bm25_results = bm25_retriever.retrieve(query=query_text, top_k=max_k)

        dense_recall: dict[int, float] = {}
        bm25_recall: dict[int, float] = {}
        dense_retrieved_docs: dict[int, list[str]] = {}
        bm25_retrieved_docs: dict[int, list[str]] = {}

        for k in k_values:
            d_rec = calculate_recall_at_k(dense_results, relevant_doc_ids, k=k)
            b_rec = calculate_recall_at_k(bm25_results, relevant_doc_ids, k=k)

            dense_recall[k] = d_rec
            bm25_recall[k] = b_rec

            dense_recall_sums[k] += d_rec
            bm25_recall_sums[k] += b_rec

            # Track unique parent doc IDs seen in top-k
            dense_retrieved_docs[k] = [r.doc_id for r in dense_results[:k]]
            bm25_retrieved_docs[k] = [r.doc_id for r in bm25_results[:k]]

        query_records.append(
            QueryEvaluationRecord(
                query_id=query_id,
                query=query_text,
                relevant_doc_ids=relevant_doc_ids,
                category=category,
                dense_recall_at_k=dense_recall,
                bm25_recall_at_k=bm25_recall,
                dense_retrieved_doc_ids=dense_retrieved_docs,
                bm25_retrieved_doc_ids=bm25_retrieved_docs,
            )
        )

    evaluable_queries = [q for q in queries if len(q.get("relevant_doc_ids", [])) > 0]
    num_eval = len(evaluable_queries) if evaluable_queries else num_queries

    dense_mean = {k: dense_recall_sums[k] / float(num_eval) for k in k_values}
    bm25_mean = {k: bm25_recall_sums[k] / float(num_eval) for k in k_values}

    return EvaluationReport(
        num_queries=num_queries,
        k_values=k_values,
        dense_mean_recall=dense_mean,
        bm25_mean_recall=bm25_mean,
        query_records=query_records,
    )


# ------------------------------------------------------------------------------
# Standalone CLI Benchmark Runner
# ------------------------------------------------------------------------------

def run_benchmark(
    k_values: list[int] = [1, 3, 5],
    chunk_strategy: str = "fixed",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    export_csv_path: Optional[str | Path] = "results/evaluation_results.csv",
) -> EvaluationReport:
    """
    Executes an end-to-end evaluation benchmark over the local corpus using the
    active configuration and writes/prints the report.
    """
    from config import load_config
    from rag.chunking import chunk_corpus
    from rag.indexing import build_indices
    from rag.ingestion import load_corpus
    from rag.providers import get_embedding_provider

    print("=" * 66)
    print("   RUNNING PHASE 3/5 RETRIEVAL BENCHMARK: DENSE vs. BM25")
    print("=" * 66)

    config = load_config()
    print(f"Loading embedding provider: {config.embedding.provider} ({config.embedding.model})...")
    embedding_provider = get_embedding_provider(config.embedding)

    print("Loading passage corpus...")
    passages = load_corpus(use_hf=False)
    print(f"Loaded {len(passages)} passages.")

    print(f"Chunking corpus (strategy={chunk_strategy}, size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_corpus(
        passages,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"Generated {len(chunks)} chunks.")

    print("Building Dense and BM25 indices...")
    dense_idx, bm25_idx = build_indices(chunks, embedding_provider=embedding_provider)

    dense_retriever = DenseRetriever(index=dense_idx, embedding_provider=embedding_provider)
    bm25_retriever = BM25Retriever(index=bm25_idx)

    print("Loading evaluation dataset...")
    queries = load_eval_queries()
    print(f"Loaded {len(queries)} labelled queries.")

    print("Running evaluation across retrievers...")
    report = evaluate_retrievers(
        queries=queries,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        k_values=k_values,
    )

    print("\n" + report.format_summary_table() + "\n")

    if export_csv_path:
        csv_file = report.export_csv(export_csv_path)
        print(f"Saved evaluation benchmark CSV to: {csv_file}")

    return report


if __name__ == "__main__":
    run_benchmark()
