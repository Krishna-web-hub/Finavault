"""FinVault Multi-Agent Enterprise AI - Streamlit Application

A secure, multi-agent financial RAG assistant with:
- Orchestrator -> Retriever -> Analyst -> Compliance Agent pipeline
- AES-256 Envelope Encryption & Token Budget enforcement
- Real-time Execution DAG / Trace visualization
- Ingestion Vault (PDF, DOCX, TXT, CSV) with auto-classification
- Multi-tenancy & Role-based Access Control (RBAC) simulator
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
import uuid
import streamlit as st

# Ensure src/ is on python path for clean imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Sync Streamlit secrets to os.environ so FinVault settings picks them up
if hasattr(st, "secrets"):
    for key, val in st.secrets.items():
        if isinstance(val, (str, int, float, bool)):
            os.environ[key] = str(val)

from finvault.config import settings
from finvault.models import User
from finvault.db import get_engine, init_db
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from finvault.ingestion.embeddings import LocalEmbeddingProvider
from finvault.retrieval.vector_store import QdrantStore
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.graph_store import PostgresGraphStore
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.reranker import LocalCrossEncoderReranker
from finvault.security.audit import PostgresAuditLog
from finvault.security.review_queue import PostgresReviewQueue
from finvault.agents.session import PostgresSessionStore
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.orchestrator import Orchestrator, OrchestratorResult
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.ingestion.classification import ClassificationSuggester
from finvault.ingestion.extraction import ExtractionAgent


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="FinVault AI | Enterprise Multi-Agent Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and dark glassmorphic look
st.markdown(
    """
    <style>
    /* Main container and typography */
    .stApp {
        background-color: #0B1120;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    /* Security Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #059669; }
    .badge-blue { background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid #2563EB; }
    .badge-purple { background-color: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid #7C3AED; }
    .badge-amber { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid #D97706; }
    .badge-red { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #DC2626; }

    /* Execution Step Card */
    .step-card {
        background: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .step-card-success { border-left-color: #10B981; }
    .step-card-warning { border-left-color: #F59E0B; }
    .step-card-error { border-left-color: #EF4444; }

    /* Citation Box */
    .citation-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing FinVault Multi-Agent Core Engine...")
def init_finvault_backend():
    """Initializes the backend infrastructure singletons (DB, Vector Store, Key Provider, Encryptor, Models)."""
    try:
        engine = get_engine()
        init_db(engine)
    except Exception as e:
        # Fallback to local SQLite if PostgreSQL is not reachable in development
        st.sidebar.warning(f"Postgres not connected ({e}). Using local SQLite database.")
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///finvault_local.db")
        init_db(engine)

    # Master Key and Envelope Encryption
    key_path = Path(".secrets/master.key")
    key_provider = LocalKeyProvider(key_path)
    encryptor = EnvelopeEncryptor(key_provider)

    # Embedding provider (Sentence Transformers local)
    embedding_provider = LocalEmbeddingProvider(settings.finvault_embedding_model)

    # Vector Store (Qdrant Cloud or Local/In-Memory)
    try:
        vector_store = QdrantStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=embedding_provider.dimension,
        )
    except Exception as e:
        st.sidebar.warning(f"Qdrant connection failed ({e}). Using in-memory vector storage.")
        vector_store = QdrantStore(
            url=":memory:",
            collection="finvault_chunks",
            dimension=embedding_provider.dimension,
        )

    # Audit, Queue, Stores
    audit_log = PostgresAuditLog(engine)
    session_store = PostgresSessionStore(engine)
    review_queue = PostgresReviewQueue(engine)
    graph_store = PostgresGraphStore(engine)
    graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)

    reranker = (
        LocalCrossEncoderReranker(settings.finvault_cross_encoder_model)
        if settings.finvault_enable_cross_encoder_rerank
        else None
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        reranker=reranker,
    )

    classification_suggester = ClassificationSuggester(embedding_provider)
    extraction_agent = ExtractionAgent()

    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        db_engine=engine,
        classification_suggester=classification_suggester,
        extraction_agent=extraction_agent,
        graph_store=graph_store,
    )

    return {
        "engine": engine,
        "encryptor": encryptor,
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "retriever": retriever,
        "audit_log": audit_log,
        "session_store": session_store,
        "review_queue": review_queue,
        "graph_retriever": graph_retriever,
        "ingestion_pipeline": ingestion_pipeline,
    }


# Initialize singletons
backend = init_finvault_backend()


# --- Sidebar: User Context, Clearance & Security Simulator ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FinVault Security Hub")
    st.caption("Enterprise Zero-Trust Multi-Agent Guard")

    st.markdown("---")
    st.subheader("👤 User Identity & RBAC")

    user_id = st.text_input("User ID / Email", value="analyst_jane@acme-corp.com")
    user_role = st.selectbox(
        "User Role",
        options=["analyst", "compliance_officer", "executive", "auditor", "guest"],
        index=0,
    )
    user_clearance = st.selectbox(
        "Clearance Level",
        options=["confidential", "internal", "public", "restricted"],
        index=0,
        help="Controls the classification ceiling of chunks this user is allowed to retrieve.",
    )
    org_id = st.text_input("Tenant / Organization ID", value="org_default")

    # Construct the User object
    current_user = User(
        id=user_id,
        role=user_role,
        clearance=user_clearance,
        org_id=org_id,
    )

    st.markdown("---")
    st.subheader("⚙️ Model & Route")
    st.info(f"**Route:** `{settings.finvault_llm_route}`\n\n**Model:** `{settings.finvault_model}`")

    st.markdown("---")
    st.subheader("🛡️ Security Controls")
    st.markdown(
        """
        - <span class="badge badge-green">AES-256 GCM</span> **Active**
        - <span class="badge badge-blue">Prompt Guard</span> **Active**
        - <span class="badge badge-purple">Compliance Veto</span> **Enforced**
        - <span class="badge badge-amber">Token Budget</span> **40k Max**
        """,
        unsafe_allow_html=True,
    )


# --- Main Header ---
st.markdown(
    f"""
    <div class="main-header">
        <h1 style="margin:0; font-size: 1.8rem; color: #FFFFFF;">🛡️ FinVault Multi-Agent Enterprise Platform</h1>
        <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
            Zero-Trust RAG Platform with AES-256 Envelope Encryption, Autonomous Agent Orchestration & Strict Compliance Verification.
        </p>
        <div style="margin-top: 12px;">
            <span class="badge badge-blue">Tenant: {org_id}</span>
            <span class="badge badge-purple">Role: {user_role.upper()}</span>
            <span class="badge badge-green">Clearance: {user_clearance.upper()}</span>
            <span class="badge badge-amber">User: {user_id}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Navigation Tabs ---
tab_chat, tab_vault, tab_compliance, tab_architecture = st.tabs(
    ["💬 Multi-Agent Chat", "📁 Document Ingestion Vault", "🔍 Audit & Compliance", "📐 Architecture & Security"]
)


# ==========================================
# TAB 1: MULTI-AGENT CHAT
# ==========================================
with tab_chat:
    st.subheader("💬 Ask Financial & Compliance Questions")
    st.caption("Every query is analyzed by Orchestrator -> Retriever -> Financial Analyst and checked by Compliance Agent.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])
            if "execution_steps" in msg and msg["execution_steps"]:
                with st.expander("🔬 View Multi-Agent Execution Steps & Citations", expanded=False):
                    for step in msg["execution_steps"]:
                        status_class = "step-card-success" if step.get("status") == "success" else "step-card-warning"
                        st.markdown(
                            f"""
                            <div class="step-card {status_class}">
                                <strong>Agent:</strong> <code>{step.get('agent_name', 'agent')}</code> | 
                                <strong>Action:</strong> <code>{step.get('name', 'step')}</code><br/>
                                <span style="color: #94A3B8; font-size: 0.8rem;">{step.get('output_preview', '')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    if msg.get("citations"):
                        st.markdown("##### 📚 Grounded Citations:")
                        for c in msg["citations"]:
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    📄 <strong>Document:</strong> {c.get('document_id', 'doc')} | 
                                    <strong>Chunk:</strong> #{c.get('chunk_index', 0)}<br/>
                                    <em>"{c.get('quote', '')}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about financial reports, AML/KYC policies, quarterly numbers..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_prompt)

        # Build orchestrator for this turn
        compliance_agent = ComplianceAgent()
        orchestrator = Orchestrator(
            retriever=backend["retriever"],
            user=current_user,
            audit_log=backend["audit_log"],
            compliance_agent=compliance_agent,
            session_store=backend["session_store"],
            review_queue=backend["review_queue"],
            graph_retriever=backend["graph_retriever"],
        )

        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("🤖 Multi-Agent Orchestration in progress (Retriever -> Analyst -> Compliance)..."):
                try:
                    res: OrchestratorResult = orchestrator.handle(
                        user_prompt,
                        session_id=st.session_state.session_id,
                    )

                    # Handle blocked / vetoed answers
                    if res.blocked:
                        st.error(f"🛑 **Query Blocked by Compliance Guardrails:** {res.block_reason}")
                        st.warning(res.answer)
                        response_content = f"🛑 **[BLOCKED - {res.block_reason}]**\n\n{res.answer}"
                    else:
                        st.markdown(res.answer)
                        response_content = res.answer

                    # Show execution steps
                    steps_data = []
                    if res.execution_steps:
                        with st.expander("🔬 View Agent Execution Trace & Citations", expanded=True):
                            for step in res.execution_steps:
                                step_dict = {
                                    "name": step.name,
                                    "agent_name": step.agent_name,
                                    "status": step.status.value if hasattr(step.status, "value") else str(step.status),
                                    "output_preview": step.output_preview,
                                }
                                steps_data.append(step_dict)
                                st.markdown(
                                    f"""
                                    <div class="step-card step-card-success">
                                        <strong>Agent:</strong> <code>{step.agent_name}</code> | 
                                        <strong>Step:</strong> <code>{step.name}</code><br/>
                                        <span style="color: #CBD5E1; font-size: 0.82rem;">{step.output_preview}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            if res.citations:
                                st.markdown("##### 📚 Grounded Citations:")
                                for c in res.citations:
                                    st.markdown(
                                        f"""
                                        <div class="citation-card">
                                            📄 <strong>Document ID:</strong> {c.document_id} | 
                                            <strong>Chunk #{c.chunk_index}</strong> | 
                                            <strong>Classification:</strong> <span class="badge badge-blue">{c.classification}</span><br/>
                                            <span style="color: #94A3B8;">Quote: <em>"{c.quote}"</em></span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "execution_steps": steps_data,
                        "citations": [
                            {"document_id": c.document_id, "chunk_index": c.chunk_index, "quote": c.quote}
                            for c in res.citations
                        ],
                    })

                except Exception as e:
                    st.error(f"❌ Execution Error: {str(e)}")


# ==========================================
# TAB 2: DOCUMENT INGESTION VAULT
# ==========================================
with tab_vault:
    st.subheader("📁 Ingest & Encrypt Financial Documents")
    st.caption("Documents are split into chunks, encrypted with AES-256 envelope encryption, and indexed into Qdrant.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Document (PDF, DOCX, TXT, CSV)",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=True,
        )

        doc_classification = st.selectbox(
            "Select Document Classification",
            options=["confidential", "internal", "public", "restricted"],
            index=0,
            help="Restricted documents cannot be externalized to third-party LLMs without explicit authorization.",
        )

        enable_graph = st.checkbox("Extract Knowledge Graph Entities", value=True)

        if st.button("🚀 Ingest & Encrypt into Vault", type="primary", disabled=not uploaded_files):
            for uploaded_file in uploaded_files:
                with st.status(f"Ingesting '{uploaded_file.name}'...", expanded=True) as status:
                    st.write("1. Reading file bytes & extracting text...")
                    file_bytes = uploaded_file.read()

                    st.write("2. Chunking & performing AES-256 Envelope Encryption (DEK/KEK)...")
                    st.write("3. Generating dense vector embeddings via BAAI/bge-small-en-v1.5...")
                    
                    try:
                        res = backend["ingestion_pipeline"].ingest_file(
                            filename=uploaded_file.name,
                            content_bytes=file_bytes,
                            org_id=org_id,
                            actor_id=user_id,
                            classification=doc_classification,
                            enable_extraction=enable_graph,
                        )
                        status.update(
                            label=f"✅ '{uploaded_file.name}' Ingested Successfully! ({res.chunk_count} chunks indexed)",
                            state="complete",
                        )
                        st.success(
                            f"**Document ID:** `{res.document_id}`\n\n"
                            f"**Chunks:** {res.chunk_count} | **Classification:** `{res.classification}` | **Encrypted:** Yes (AES-256-GCM)"
                        )
                    except Exception as e:
                        status.update(label=f"❌ Failed to ingest '{uploaded_file.name}'", state="error")
                        st.error(f"Error: {e}")

    with col2:
        st.markdown("#### 📦 Sample Financial Documents")
        st.markdown("Quickly test ingestion with built-in sample enterprise files:")
        sample_docs_dir = ROOT_DIR / "sample_docs"
        if sample_docs_dir.exists():
            sample_files = list(sample_docs_dir.glob("*.txt")) + list(sample_docs_dir.glob("*.csv"))
            for sf in sample_files:
                if st.button(f"📥 Ingest Sample: {sf.name}", key=f"sample_{sf.name}"):
                    with st.spinner(f"Ingesting {sf.name}..."):
                        content = sf.read_bytes()
                        res = backend["ingestion_pipeline"].ingest_file(
                            filename=sf.name,
                            content_bytes=content,
                            org_id=org_id,
                            actor_id=user_id,
                            classification="confidential",
                        )
                        st.success(f"Ingested `{sf.name}` ({res.chunk_count} chunks)")


# ==========================================
# TAB 3: AUDIT & COMPLIANCE
# ==========================================
with tab_compliance:
    st.subheader("🔍 Tamper-Evident Audit Logs & Compliance Review")
    st.caption("Every retrieval, prompt externalization, decryption, and agent decision is logged with cryptographic binding.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Active Encryption Algorithm", value="AES-256-GCM")
    with col_b:
        st.metric(label="Compliance Policy Enforcer", value="Strict Grounding Veto")

    st.markdown("---")
    st.markdown("#### 📋 Recent Audit Trail")
    try:
        from sqlalchemy import text
        with backend["engine"].connect() as conn:
            logs = conn.execute(
                text("SELECT timestamp, actor, action, resource, details FROM audit_log ORDER BY timestamp DESC LIMIT 25")
            ).fetchall()
            if logs:
                st.table([
                    {
                        "Timestamp": str(r[0]),
                        "Actor": r[1],
                        "Action": r[2],
                        "Resource": r[3],
                        "Details": str(r[4])[:80] + "...",
                    }
                    for r in logs
                ])
            else:
                st.info("No audit logs recorded yet in this session.")
    except Exception as e:
        st.info(f"Audit log query: {e}")


# ==========================================
# TAB 4: ARCHITECTURE & ZERO-TRUST
# ==========================================
with tab_architecture:
    st.subheader("📐 System Architecture & Threat Model")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            User([Enterprise User]) -->|Clearance + Org Context| Guard[Guardrail & Scanner]
            Guard -->|Sanitized Question| Orch[Orchestrator Agent]
            
            subgraph Agent Core
                Orch -->|Multi-Round Search| Retr[Retriever Agent]
                Retr -->|Vector + BM25 + Cross-Encoder| Rerank[Reranker]
                Rerank -->|Encrypted Chunks| Decrypt[Envelope Decryption AES-256]
                Decrypt -->|Plaintext Chunks| Analyst[Financial Analyst Agent]
            end
            
            Analyst -->|Draft Analysis + Citations| Comp[Compliance Agent]
            Comp -->|Verify Grounding & Classifications| FinalResult{Verdict}
            FinalResult -->|Approved| User
            FinalResult -->|Vetoed| BlockedNotice[Blocked / Security Review Queue]
        ```
        """,
        unsafe_allow_html=False,
    )
    st.markdown(
        """
        ### 🛡️ Core Security Invariants:
        1. **Envelope Encryption**: Document chunks are stored strictly as ciphertext + wrapped DEK in the vector database.
        2. **Multi-hop Grounding**: The Analyst agent generates verifiable citations; the Compliance agent acts as a fail-closed veto if claims are ungrounded.
        3. **Data Classification Barrier**: Documents marked `restricted` are filtered at the retrieval gateway and never sent to external LLMs.
        4. **Token Budget Enforcement**: 40k ceiling per request prevents runaway recursion and denial-of-wallet risks.
        """
    )
