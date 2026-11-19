"""
RAG Pipeline package for Yotta Workstream A.
"""

from rag.chunking import Chunk, FixedSizeChunker, SentenceBoundaryChunker, chunk_corpus, chunk_document
from rag.evaluation import EvaluationReport, calculate_recall_at_k, evaluate_retrievers, load_eval_queries
from rag.generation import GenerationResult, GroundedGenerator, generate_grounded_answer
from rag.grading import GradingDecision, GradingResult, RelevanceGrader, grade_relevance
from rag.indexing import BM25Index, DenseIndex, build_bm25_index, build_dense_index, build_indices
from rag.ingestion import Passage, clean_and_validate_passages, clean_passage_text, load_corpus
from rag.pipeline import RAGPipeline, RAGResponse, create_pipeline
from rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RetrievalResult, retrieve_top_k

__version__ = "0.1.0"

__all__ = [
    # Ingestion
    "Passage",
    "clean_passage_text",
    "clean_and_validate_passages",
    "load_corpus",
    # Chunking
    "Chunk",
    "FixedSizeChunker",
    "SentenceBoundaryChunker",
    "chunk_document",
    "chunk_corpus",
    # Indexing
    "DenseIndex",
    "BM25Index",
    "build_dense_index",
    "build_bm25_index",
    "build_indices",
    # Retrieval
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "retrieve_top_k",
    # Grading (Phase 4)
    "GradingDecision",
    "GradingResult",
    "RelevanceGrader",
    "grade_relevance",
    # Generation (Phase 4)
    "GenerationResult",
    "GroundedGenerator",
    "generate_grounded_answer",
    # Pipeline (Phase 4)
    "RAGPipeline",
    "RAGResponse",
    "create_pipeline",
    # Evaluation
    "calculate_recall_at_k",
    "evaluate_retrievers",
    "load_eval_queries",
    "EvaluationReport",
]
