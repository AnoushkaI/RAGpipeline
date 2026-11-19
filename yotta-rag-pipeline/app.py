"""
Streamlit UI for Yotta Workstream A RAG Pipeline.

Phase 1 provides:
- Live Configuration Overview (safe masked summary).
- Provider health check and smoke-test status.
- Roadmap status for upcoming phases.
"""

from __future__ import annotations

import streamlit as st
from config import load_config
from rag.providers import get_embedding_provider, get_llm_provider

st.set_page_config(
    page_title="Yotta RAG Pipeline",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Yotta Workstream A — RAG Pipeline")
st.caption("Implementation Blueprint • Phase 1: Foundation & Provider Setup")

config = load_config()

with st.sidebar:
    st.header("⚙️ Configuration")
    summary = config.get_summary()
    st.subheader("LLM Generation")
    st.write(f"**Provider:** `{summary['llm']['provider']}`")
    st.write(f"**Model:** `{summary['llm']['model']}`")
    st.write(f"**Base URL:** `{summary['llm']['base_url']}`")
    st.write(f"**API Key:** `{summary['llm']['api_key']}`")

    st.subheader("Dense Embeddings")
    st.write(f"**Provider:** `{summary['embedding']['provider']}`")
    st.write(f"**Model:** `{summary['embedding']['model']}`")
    st.write(f"**Base URL:** `{summary['embedding']['base_url']}`")
    st.write(f"**API Key:** `{summary['embedding']['api_key']}`")

    st.subheader("RAG Parameters")
    st.write(f"**Chunk Size:** `{summary['rag']['chunk_size']}`")
    st.write(f"**Chunk Overlap:** `{summary['rag']['chunk_overlap']}`")
    st.write(f"**Top-K:** `{summary['rag']['top_k']}`")
    st.write(f"**Threshold:** `{summary['rag']['relevance_threshold']}`")
    st.write(f"**Max Retries:** `{summary['rag']['max_retries']}`")

st.header("1. Provider Health & Connectivity")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Generation Provider")
    llm_prov = get_llm_provider(config.llm)
    llm_health = llm_prov.health_check()
    if llm_health.status == "OK":
        st.success(f"Status: {llm_health.status} ({llm_health.latency_ms:.1f}ms)\n\n{llm_health.message}")
    elif llm_health.status == "MISSING_CREDENTIALS":
        st.warning(f"Status: {llm_health.status}\n\n{llm_health.message}")
    elif llm_health.status == "UNTESTED":
        st.info(f"Status: {llm_health.status}\n\n{llm_health.message}")
    else:
        st.error(f"Status: {llm_health.status}\n\n{llm_health.message}")

with col2:
    st.subheader("Embedding Provider")
    emb_prov = get_embedding_provider(config.embedding)
    emb_health = emb_prov.health_check()
    if emb_health.status == "OK":
        st.success(f"Status: {emb_health.status} ({emb_health.latency_ms:.1f}ms)\n\n{emb_health.message}")
    elif emb_health.status == "MISSING_CREDENTIALS":
        st.warning(f"Status: {emb_health.status}\n\n{emb_health.message}")
    elif emb_health.status == "UNTESTED":
        st.info(f"Status: {emb_health.status}\n\n{emb_health.message}")
    else:
        st.error(f"Status: {emb_health.status}\n\n{emb_health.message}")

st.header("2. Pipeline Implementation Roadmap")
roadmap = [
    ("Phase 1: Foundation & Provider Setup", "Completed (Modular structure, provider abstraction, health checks)", "✅"),
    ("Phase 2: MS MARCO Ingestion & Chunking", "Pending approval (MS MARCO subset, JSONL fallback, 2 chunking strategies)", "⏳"),
    ("Phase 3: Dense & BM25 Indexing / Retrieval", "Pending Phase 2 (Cosine similarity + rank-bm25 comparison)", "⏳"),
    ("Phase 4: Relevance Grading & Grounded Generation", "Pending Phase 3 (Threshold/LLM grading, 1 rewrite retry)", "⏳"),
    ("Phase 5: Evaluation & Full UI Integration", "Pending Phase 4 (Recall@K, faithfulness, latency, end-to-end interactive demo)", "⏳"),
]

for title, desc, icon in roadmap:
    st.write(f"**{icon} {title}** — {desc}")
