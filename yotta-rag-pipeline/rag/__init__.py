"""
RAG Pipeline package for Yotta Workstream A.
"""

from rag.chunking import Chunk, FixedSizeChunker, SentenceBoundaryChunker, chunk_corpus, chunk_document
from rag.evaluation import EvaluationReport, calculate_recall_at_k, evaluate_retrievers, load_eval_queries
from rag.indexing import BM25Index, DenseIndex, build_bm25_index, build_dense_index, build_indices
from rag.ingestion import Passage, clean_and_validate_passages, clean_passage_text, load_corpus
from rag.retrieval import BM25Retriever, DenseRetriever, RetrievalResult, retrieve_top_k

__version__ = "0.1.0"

__all__ = [
    "Passage",
    "clean_passage_text",
    "clean_and_validate_passages",
    "load_corpus",
    "Chunk",
    "FixedSizeChunker",
    "SentenceBoundaryChunker",
    "chunk_document",
    "chunk_corpus",
    "DenseIndex",
    "BM25Index",
    "build_dense_index",
    "build_bm25_index",
    "build_indices",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "retrieve_top_k",
    "calculate_recall_at_k",
    "evaluate_retrievers",
    "load_eval_queries",
    "EvaluationReport",
]
