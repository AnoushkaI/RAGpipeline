"""
Unit tests for Provider Abstraction, adapters, error handling, and health checks.
"""

from __future__ import annotations

import pytest
from config import EmbeddingConfig, LLMConfig
from rag.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    MockEmbeddingProvider,
    MockLLMProvider,
    ModelProvider,
    NvidiaNimEmbeddingProvider,
    NvidiaNimLLMProvider,
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
    OpenAILLMProvider,
    ProviderConfigurationError,
    ProviderConnectionError,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
    get_llm_provider,
)


def test_mock_llm_and_embedding():
    """Verify mock providers work deterministically."""
    mock_llm = MockLLMProvider(model="test-llm", default_response="Grounded Answer.")
    resp = mock_llm.generate("Hello?")
    assert resp == "Grounded Answer."
    health = mock_llm.health_check()
    assert health.status == "OK"

    mock_emb = MockEmbeddingProvider(dimension=8)
    vectors = mock_emb.embed(["chunk one", "chunk two"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 8
    health_emb = mock_emb.health_check()
    assert health_emb.status == "OK"


def test_openai_missing_credentials_handling():
    """Verify OpenAI provider gracefully reports missing credentials."""
    cfg = LLMConfig(provider="openai", api_key=None)
    prov = OpenAILLMProvider(cfg)
    health = prov.health_check()
    assert health.status == "MISSING_CREDENTIALS"

    with pytest.raises(ProviderConfigurationError):
        prov.generate("test prompt")


def test_nvidia_nim_missing_credentials_handling():
    """Verify NVIDIA NIM provider gracefully reports missing credentials."""
    cfg = LLMConfig(provider="nvidia_nim", api_key=None)
    prov = NvidiaNimLLMProvider(cfg)
    health = prov.health_check()
    assert health.status == "MISSING_CREDENTIALS"

    with pytest.raises(ProviderConfigurationError):
        prov.generate("test prompt")


def test_ollama_unreachable_endpoint():
    """Verify Ollama provider gracefully handles unreachable local daemon."""
    cfg = LLMConfig(provider="ollama", base_url="http://127.0.0.1:9999", timeout=1)
    prov = OllamaLLMProvider(cfg)
    health = prov.health_check()
    assert health.status == "FAILED"
    assert "Cannot reach Ollama" in health.message

    with pytest.raises(ProviderConnectionError):
        prov.generate("test")


def test_sentence_transformer_local_provider():
    """Verify local SentenceTransformers provider generates embeddings and handles health checks."""
    import numpy as np

    class FakeSTModel:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 384), dtype=np.float32)

    cfg = EmbeddingConfig(provider="sentence_transformers", model="all-MiniLM-L6-v2")
    prov = SentenceTransformerEmbeddingProvider(cfg)
    
    # Inject fake model to avoid 2-minute cold-load PyTorch DLL initialization during fast unit tests
    prov._model_instance = FakeSTModel()
    
    health = prov.health_check()
    assert health.status == "OK"
    assert health.latency_ms is not None

    embeddings = prov.embed(["Machine learning is fascinating.", "Information retrieval."])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


def test_model_provider_facade():
    """Verify unified ModelProvider delegates to LLM and Embedding implementations."""
    llm = MockLLMProvider(default_response="Mocked response")
    emb = MockEmbeddingProvider(dimension=16)
    model_provider = ModelProvider(llm=llm, embedding=emb)

    assert model_provider.generate("test") == "Mocked response"
    vectors = model_provider.embed(["test text"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 16

    health_map = model_provider.run_health_checks()
    assert health_map["llm"].status == "OK"
    assert health_map["embedding"].status == "OK"


def test_factory_functions():
    """Verify provider factories correctly instantiate requested classes."""
    llm = get_llm_provider(LLMConfig(provider="mock"))
    assert isinstance(llm, BaseLLMProvider)

    emb = get_embedding_provider(EmbeddingConfig(provider="mock"))
    assert isinstance(emb, BaseEmbeddingProvider)

    with pytest.raises(ProviderConfigurationError):
        get_llm_provider(LLMConfig(provider="nonexistent"))


def test_nvidia_nim_staging_prefix_and_error_sanitization():
    """Verify candidate model IDs include staging prefixes and secrets are masked."""
    cfg = LLMConfig(provider="nvidia_nim", model="nvidia/nemotron-3.5-bolt-e4b", api_key="nvapi-super-secret-123")
    prov = NvidiaNimLLMProvider(cfg)
    
    candidates = prov._candidate_models()
    assert candidates == ["nvidia/nemotron-3.5-bolt-e4b", "stg/nvidia/nemotron-3.5-bolt-e4b"]

    cfg_stg = LLMConfig(provider="nvidia_nim", model="stg/nvidia/nemotron-3.5-bolt-e4b", api_key="nvapi-super-secret-123")
    prov_stg = NvidiaNimLLMProvider(cfg_stg)
    assert prov_stg._candidate_models() == ["stg/nvidia/nemotron-3.5-bolt-e4b", "nvidia/nemotron-3.5-bolt-e4b"]

    # Error sanitization
    raw_error = "Error from https://integrate.api.nvidia.com with nvapi-super-secret-123 and Bearer secret-token"
    safe_error = prov._sanitize_error(raw_error)
    assert "super-secret" not in safe_error
    assert "secret-token" not in safe_error
    assert "***" in safe_error
    assert "Bearer ***" in safe_error


def test_nvidia_nim_embedding_model_separation():
    """Verify Nemotron generation model is rejected if mistakenly configured as embedding model."""
    from rag.providers import NvidiaNimEmbeddingProvider
    cfg = EmbeddingConfig(provider="nvidia_nim", model="nvidia/nemotron-3.5-bolt-e4b", api_key="nvapi-test")
    prov = NvidiaNimEmbeddingProvider(cfg)
    health = prov.health_check()
    assert health.status == "FAILED"
    assert "LLM text generation model, not an embedding model" in health.message

    with pytest.raises(ProviderConfigurationError):
        prov.embed(["test text"])


def test_ollama_embed_endpoint_batching(monkeypatch):
    """Verify Ollama embedding provider calls /api/embed with batched input and parses response."""
    from rag.providers import OllamaEmbeddingProvider
    import requests

    called_urls = []
    called_payloads = []

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        def raise_for_status(self):
            pass

    def fake_post(url, json=None, **kwargs):
        called_urls.append(url)
        called_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    cfg = EmbeddingConfig(provider="ollama", model="nomic-embed-text")
    prov = OllamaEmbeddingProvider(cfg)
    vectors = prov.embed(["passage one", "passage two"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert any("/api/embed" in u for u in called_urls)
    assert called_payloads[0]["input"] == ["passage one", "passage two"]


def test_ollama_missing_model_honesty(monkeypatch):
    """Verify Ollama health check reports FAILED when configured model is not installed."""
    import requests

    class FakeTagsResponse:
        status_code = 200
        def json(self):
            # Server is running, but only has llama2, not nomic-embed-text
            return {"models": [{"name": "llama2:latest"}]}
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeTagsResponse())

    cfg = EmbeddingConfig(provider="ollama", model="nomic-embed-text")
    prov = OllamaEmbeddingProvider(cfg)
    health = prov.health_check()
    assert health.status == "FAILED"
    assert "not installed" in health.message
    assert "ollama pull nomic-embed-text" in health.message


def test_smoke_test_honesty():
    """Verify smoke test returns failure (code 1) when credentials or models are missing."""
    from rag.providers import run_provider_smoke_test
    from config import AppConfig, LLMConfig

    # OpenAI selected with missing credentials (using mock embedding to avoid heavy ST model loading in fast unit test)
    cfg = AppConfig(
        llm=LLMConfig(provider="openai", api_key=None),
        embedding=EmbeddingConfig(provider="mock"),
    )
    code = run_provider_smoke_test(cfg)
    assert code == 1

    # Mock mode with everything present returns success (code 0)
    cfg_mock = AppConfig(llm=LLMConfig(provider="mock"), embedding=EmbeddingConfig(provider="mock"))
    code_mock = run_provider_smoke_test(cfg_mock)
    assert code_mock == 0

