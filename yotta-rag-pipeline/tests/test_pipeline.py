"""
Integration and Unit Tests for End-to-End RAG Pipeline (Phase 4).

Tests cover:
1. Successful end-to-end question answering and source document citation.
2. Irrelevant query fallback, query rewriting, and graceful abstention.
3. Empty/whitespace query validation.
4. Empty index handling.
5. Multi-strategy retrieval (dense, bm25, hybrid).
6. Relevance grading methods (threshold vs LLM).
7. Provider failure / exception resilience.
8. Bounded retry mechanism (strictly 1 retry max).
9. Grounded citation validation filtering hallucinated document IDs.
"""

from __future__ import annotations

import pytest

from config import AppConfig, LLMConfig, EmbeddingConfig, RAGConfig
from rag.chunking import Chunk
from rag.generation import GroundedGenerator, generate_grounded_answer
from rag.grading import RelevanceGrader, grade_relevance
from rag.ingestion import Passage
from rag.pipeline import RAGPipeline, RAGResponse, create_pipeline
from rag.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    MockEmbeddingProvider,
    MockLLMProvider,
    ProviderExecutionError,
    ProviderHealth,
)
from rag.retrieval import RetrievalResult


# ------------------------------------------------------------------------------
# Test Fixtures & Helpers
# ------------------------------------------------------------------------------

def make_test_passages() -> list[Passage]:
    return [
        Passage(
            doc_id="doc_001",
            text="Backpropagation calculates gradients of the loss function with respect to weights using the chain rule.",
            metadata={"topic": "neural_networks"},
        ),
        Passage(
            doc_id="doc_002",
            text="B-Tree indexes maintain sorted key-value pairs in balanced multi-way tree nodes for fast disk lookups.",
            metadata={"topic": "databases"},
        ),
        Passage(
            doc_id="doc_003",
            text="The TCP handshake establishes reliable network connections between client and server via SYN, SYN-ACK, ACK.",
            metadata={"topic": "networking"},
        ),
    ]


class FlakyOrFailingLLMProvider(BaseLLMProvider):
    """LLM provider simulating runtime failure / timeout."""
    def generate(self, prompt: str) -> str:
        raise ProviderExecutionError("NVIDIA NIM Gateway 504 Gateway Timeout")

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(name="FlakyLLM", kind="generation", status="FAILED", model="flaky")


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_pipeline_successful_generation_and_citation():
    """Verify end-to-end question answering produces grounded answer and source citation."""
    mock_llm = MockLLMProvider(
        default_response="Backpropagation computes gradients using the chain rule [doc_001]."
    )
    mock_embed = MockEmbeddingProvider(dimension=8)
    config = AppConfig(
        rag=RAGConfig(chunk_size=150, chunk_overlap=20, top_k=2, relevance_threshold=0.1)
    )

    pipeline = RAGPipeline(
        config=config,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        passages=make_test_passages(),
    )

    response = pipeline.run("How does backpropagation compute gradients?", strategy="dense")

    assert isinstance(response, RAGResponse)
    assert response.has_sufficient_context is True
    assert "Backpropagation" in response.answer
    assert "doc_001" in response.cited_doc_ids
    assert len(response.passed_chunks) > 0
    assert len(response.source_passages) == len(response.passed_chunks)
    assert response.strategy == "dense"
    assert response.latency_ms >= 0.0
    assert response.timing == response.latency_ms
    assert response.retry_flag is False
    assert response.grading_method == "threshold"
    assert "decisions" in response.grading_result

    # Verify dict conversion and item access
    res_dict = response.to_dict()
    assert res_dict["query"] == "How does backpropagation compute gradients?"
    assert "source_passages" in res_dict
    assert "grading_method" in res_dict
    assert "grading_result" in res_dict
    assert "timing" in res_dict
    assert response["answer"] == response.answer


def test_pipeline_irrelevant_query_triggers_retry_and_abstains():
    """Verify an irrelevant query triggers query rewrite and abstains gracefully without hallucination."""
    mock_llm = MockLLMProvider(default_response="rewritten query terms")
    mock_embed = MockEmbeddingProvider(dimension=8)

    config = AppConfig(
        rag=RAGConfig(chunk_size=150, chunk_overlap=20, top_k=2, relevance_threshold=0.5, max_retries=1)
    )

    pipeline = RAGPipeline(
        config=config,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        passages=make_test_passages(),
    )

    # BM25 has zero term overlap with "Quantum entanglement qubit superposition"
    response = pipeline.run("Quantum entanglement qubit superposition", strategy="bm25")

    assert response.has_sufficient_context is False
    assert response.retry_attempted is True
    assert response.rewritten_query == "rewritten query terms"
    assert "cannot answer" in response.answer.lower()
    assert len(response.cited_doc_ids) == 0
    assert len(response.passed_chunks) == 0


def test_pipeline_dense_insufficient_threshold_retry():
    """Verify that dense retrieval below threshold triggers retry and abstains."""
    mock_llm = MockLLMProvider(default_response="rewritten query")
    mock_embed = MockEmbeddingProvider(dimension=8)

    # Threshold higher than MockEmbedding dot product (~0.9987)
    config = AppConfig(
        rag=RAGConfig(chunk_size=150, chunk_overlap=20, top_k=2, relevance_threshold=0.9999, max_retries=1)
    )

    pipeline = RAGPipeline(
        config=config,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        passages=make_test_passages(),
    )

    response = pipeline.run("Quantum computing entanglement", strategy="dense")
    assert response.has_sufficient_context is False
    assert response.retry_attempted is True
    assert "cannot answer" in response.answer.lower()


def test_pipeline_empty_query():
    """Verify empty or whitespace queries are caught immediately with no unnecessary calls."""
    pipeline = RAGPipeline(
        llm_provider=MockLLMProvider(),
        embedding_provider=MockEmbeddingProvider(),
        passages=make_test_passages(),
    )

    res_empty = pipeline.run("")
    assert res_empty.has_sufficient_context is False
    assert "valid, non-empty query" in res_empty.answer

    res_spaces = pipeline.run("    ")
    assert res_spaces.has_sufficient_context is False
    assert "valid, non-empty query" in res_spaces.answer


def test_pipeline_empty_index():
    """Verify pipeline handles unindexed state safely."""
    pipeline = RAGPipeline(
        llm_provider=MockLLMProvider(),
        embedding_provider=MockEmbeddingProvider(),
        passages=[],  # No passages
    )

    response = pipeline.run("What is backpropagation?", strategy="dense")
    assert response.has_sufficient_context is False
    assert "No documents have been indexed" in response.answer


def test_pipeline_retrieval_strategies():
    """Verify dense, bm25, and hybrid strategies all execute cleanly."""
    pipeline = RAGPipeline(
        config=AppConfig(rag=RAGConfig(relevance_threshold=0.0, top_k=2)),
        llm_provider=MockLLMProvider(default_response="Grounded answer [doc_002]."),
        embedding_provider=MockEmbeddingProvider(),
        passages=make_test_passages(),
    )

    for strat in ["dense", "bm25", "hybrid"]:
        res = pipeline.run("B-Tree index lookups", strategy=strat)
        assert res.strategy == strat
        assert len(res.retrieved_chunks) > 0
        assert res.has_sufficient_context is True


def test_relevance_grader_threshold_and_llm():
    """Verify threshold and LLM grading methods evaluate candidate chunks."""
    candidates = [
        RetrievalResult(chunk_id="c1", doc_id="d1", text="Relevant text", score=0.8, strategy="dense"),
        RetrievalResult(chunk_id="c2", doc_id="d2", text="Irrelevant text", score=0.2, strategy="dense"),
    ]

    # 1. Threshold grading
    t_grader = RelevanceGrader(method="threshold", threshold=0.5)
    t_res = t_grader.grade("test query", candidates)
    assert t_res.is_sufficient is True
    assert len(t_res.passed_chunks) == 1
    assert t_res.passed_chunks[0].chunk_id == "c1"
    assert len(t_res.rejected_chunks) == 1

    # 2. LLM grading
    class MockYesNoLLM(BaseLLMProvider):
        def generate(self, prompt: str) -> str:
            return "YES" if "Relevant text" in prompt else "NO"
        def health_check(self):
            return ProviderHealth(name="mock", kind="generation", status="OK", model="m")

    llm_grader = RelevanceGrader(method="llm", llm_provider=MockYesNoLLM())
    l_res = llm_grader.grade("test query", candidates)
    assert l_res.is_sufficient is True
    assert len(l_res.passed_chunks) == 1
    assert l_res.passed_chunks[0].chunk_id == "c1"


def test_pipeline_provider_failure_resilience():
    """Verify pipeline catches LLM provider execution exceptions gracefully."""
    failing_llm = FlakyOrFailingLLMProvider()
    config = AppConfig(rag=RAGConfig(relevance_threshold=0.0, max_retries=0))

    pipeline = RAGPipeline(
        config=config,
        llm_provider=failing_llm,
        embedding_provider=MockEmbeddingProvider(),
        passages=make_test_passages(),
    )

    response = pipeline.run("Explain TCP handshake", strategy="dense")
    assert response.has_sufficient_context is False
    assert "provider error" in response.answer.lower()
    assert "Timeout" in response.answer


def test_citation_validation_filters_hallucinated_ids():
    """Verify that hallucinated citations in LLM response are discarded."""
    mock_llm = MockLLMProvider(
        default_response="Grounded text [doc_001] with hallucinated citation [doc_9999] and invalid tag [xyz]."
    )
    generator = GroundedGenerator(llm_provider=mock_llm)

    chunks = [
        RetrievalResult(chunk_id="c1", doc_id="doc_001", text="Grounded passage text", score=0.9),
    ]

    res = generator.generate("Query", chunks)
    assert res.cited_doc_ids == ["doc_001"]
    assert "doc_9999" not in res.cited_doc_ids
    assert "xyz" not in res.cited_doc_ids


def test_generator_abstention_when_empty_context():
    """Verify generator returns default abstention when no chunks are passed."""
    mock_llm = MockLLMProvider(default_response="This should never be called.")
    generator = GroundedGenerator(llm_provider=mock_llm)

    res = generator.generate("Query", [])
    assert res.has_sufficient_context is False
    assert "cannot answer" in res.answer.lower()
    assert len(res.cited_doc_ids) == 0


def test_insufficient_context_bounded_to_exactly_one_retry(monkeypatch):
    """Verify that insufficient context triggers exactly one rewrite and retry, never an unbounded loop."""
    mock_llm = MockLLMProvider(default_response="rewritten query")
    mock_embed = MockEmbeddingProvider(dimension=8)

    config = AppConfig(
        rag=RAGConfig(chunk_size=150, chunk_overlap=20, top_k=2, relevance_threshold=0.5, max_retries=10)
    )

    pipeline = RAGPipeline(
        config=config,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        passages=make_test_passages(),
    )

    rewrite_call_count = 0
    original_rewrite = pipeline.rewrite_query

    def counting_rewrite(q: str) -> str:
        nonlocal rewrite_call_count
        rewrite_call_count += 1
        return original_rewrite(q)

    monkeypatch.setattr(pipeline, "rewrite_query", counting_rewrite)

    # BM25 has zero match for this query
    response = pipeline.run("Quantum entanglement qubit superposition", strategy="bm25")

    # Strictly exactly 1 rewrite and 1 retry attempted
    assert rewrite_call_count == 1
    assert response.retry_attempted is True
    assert response.retry_flag is True
    assert response.rewritten_query == "rewritten query"
    assert response.has_sufficient_context is False
    assert "cannot answer" in response.answer.lower()


def test_only_accepted_passages_and_sources_reach_generation():
    """Verify that rejected passages and their doc IDs never reach generation."""
    captured_prompts: list[str] = []

    class PromptCapturingLLM(BaseLLMProvider):
        def generate(self, prompt: str) -> str:
            captured_prompts.append(prompt)
            return "Backpropagation uses the chain rule [doc_001]."

        def health_check(self):
            return ProviderHealth(name="capturing", kind="generation", status="OK", model="m")

    capturing_llm = PromptCapturingLLM()
    # Grader will only pass chunks with score >= 0.7
    grader = RelevanceGrader(method="threshold", threshold=0.7)

    passages = [
        Passage(doc_id="doc_001", text="Backpropagation calculates gradients of the loss function.", metadata={}),
        Passage(doc_id="doc_999", text="Unrelated text about cooking recipes and ingredients.", metadata={}),
    ]

    pipeline = RAGPipeline(
        config=AppConfig(rag=RAGConfig(relevance_threshold=0.7)),
        llm_provider=capturing_llm,
        embedding_provider=MockEmbeddingProvider(),
        grader=grader,
    )

    # Build controlled candidates
    accepted_chunk = RetrievalResult(
        chunk_id="doc_001_f0",
        doc_id="doc_001",
        text="Backpropagation calculates gradients of the loss function.",
        score=0.95,  # Passes threshold 0.7
        strategy="dense",
    )
    rejected_chunk = RetrievalResult(
        chunk_id="doc_999_f0",
        doc_id="doc_999",
        text="Unrelated text about cooking recipes and ingredients.",
        score=0.20,  # Fails threshold 0.7
        strategy="dense",
    )

    grading_result = grader.grade("Backpropagation", [accepted_chunk, rejected_chunk])
    assert len(grading_result.passed_chunks) == 1
    assert len(grading_result.rejected_chunks) == 1

    generator = GroundedGenerator(llm_provider=capturing_llm)
    gen_result = generator.generate("Backpropagation", grading_result.passed_chunks)

    assert len(captured_prompts) == 1
    prompt_text = captured_prompts[0]

    # Accepted passage and its doc_id MUST be in prompt
    assert "[doc_001]" in prompt_text
    assert "Backpropagation calculates gradients" in prompt_text

    # Rejected passage and its doc_id MUST NOT be in prompt
    assert "doc_999" not in prompt_text
    assert "cooking recipes" not in prompt_text

    # Output citations must only contain doc_001
    assert gen_result.cited_doc_ids == ["doc_001"]


def test_unsupported_question_rejection_across_strategies(monkeypatch):
    """
    Regression test: Verifies that unsupported questions (e.g. Yotta market share)
    are rejected across dense, BM25, and hybrid strategies, avoiding raw BM25 score
    comparison bugs and ensuring graceful abstention without hallucination.
    """
    from rag.ingestion import load_corpus

    class FormulatingMockLLM(BaseLLMProvider):
        def generate(self, prompt: str) -> str:
            if "reformulat" in prompt.lower() or "search query" in prompt.lower():
                return "Yotta Infrastructure market share India cloud services"
            return "Grounded answer text [doc_001]."

        def health_check(self):
            return ProviderHealth(name="mock", kind="generation", status="OK", model="m")

    class LowSimilarityEmbedding(BaseEmbeddingProvider):
        """Simulates low-similarity vectors (< 0.50) for unsupported questions."""
        def embed(self, texts: list[str]) -> list[list[float]]:
            # Map unsupported query to an orthogonal dimension (127) away from docs (0..99)
            dim = 128
            vecs = []
            for t in texts:
                vec = [0.0] * dim
                if "yotta" in t.lower() or "market share" in t.lower():
                    vec[127] = 1.0
                else:
                    vec[abs(hash(t)) % 100] = 1.0
                vecs.append(vec)
            return vecs

        def health_check(self):
            return ProviderHealth(name="low_sim", kind="embedding", status="OK", model="m")

    mock_llm = FormulatingMockLLM()
    mock_embed = LowSimilarityEmbedding()
    passages = load_corpus(use_hf=False, sample_size=20)

    config = AppConfig(
        rag=RAGConfig(chunk_size=300, chunk_overlap=50, top_k=3, relevance_threshold=0.5, max_retries=1)
    )

    pipeline = RAGPipeline(
        config=config,
        llm_provider=mock_llm,
        embedding_provider=mock_embed,
        passages=passages,
    )

    unsupported_q = "What is the current market share of Yotta in the Indian cloud computing market?"

    for strat in ["dense", "bm25", "hybrid"]:
        res = pipeline.run(unsupported_q, strategy=strat)
        assert res.has_sufficient_context is False, f"Strategy {strat} failed to reject unsupported context!"
        assert res.retry_attempted is True, f"Strategy {strat} did not attempt retry!"
        assert "cannot answer" in res.answer.lower()
        assert len(res.cited_doc_ids) == 0
        assert len(res.passed_chunks) == 0
