"""FinVault Multi-Agent Enterprise AI - Streamlit Application

A secure, multi-agent financial RAG platform featuring:
- Orchestrator -> Retriever -> Analyst -> Compliance Agent pipeline
- AES-256 Envelope Encryption (DEK/KEK) & Token Budget enforcement
- Real-time Execution DAG / Trace visualization
- Ingestion Vault (PDF, DOCX, TXT, CSV) with auto-classification & Knowledge Graph
- Tamper-Evident Hash-Chained Audit Trail & Compliance Review Queue
- Multi-tenancy & Role-based Access Control (RBAC) simulator
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# Ensure src/ is on python path for clean imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Sync Streamlit secrets to os.environ so FinVault settings picks them up.
# hasattr(st, "secrets") is not a guard: the attribute always exists, and it is
# .items() that raises StreamlitSecretNotFoundError when no secrets.toml is
# present — which is the normal case for a local run and for a Cloud app before
# any secrets are entered. Catching it keeps the no-config path alive; env vars
# and .env still supply settings on their own.
try:
    for key, val in st.secrets.items():
        if isinstance(val, (str, int, float, bool)):
            os.environ[key] = str(val)
except Exception:
    pass

from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.orchestrator import Orchestrator, OrchestratorResult
from finvault.agents.session import PostgresSessionStore
from finvault.config import settings
from finvault.db import get_engine, init_db
from finvault.ingestion.classification import ClassificationSuggester
from finvault.ingestion.embeddings import LocalEmbeddingProvider
from finvault.ingestion.extraction import ExtractionAgent
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.models import CLASSIFICATION_RANK, Classification, Role, User
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import PostgresGraphStore
from finvault.retrieval.reranker import LocalCrossEncoderReranker
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import QdrantStore
from finvault.security.access_control import check_clearance
from finvault.security.audit import PostgresAuditLog
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider
from finvault.security.review_queue import PostgresReviewQueue


def clearance_ceiling(role: Role) -> Classification:
    """Highest classification `role` is allowed to retrieve.

    Derived rather than stored: check_clearance is the authority, so this can
    never disagree with what the retrieval path will actually permit.
    """
    return max(
        (c for c in Classification if check_clearance(role, c)),
        key=lambda c: CLASSIFICATION_RANK[c],
    )


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="FinVault AI | Multi-Agent Financial Security Platform",
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
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 22px 26px;
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
        margin-bottom: 4px;
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

    /* Metric card */
    .metric-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing FinVault Multi-Agent Core Engine...")
def init_finvault_backend():
    """Initializes the backend infrastructure singletons (DB, Vector Store, Key Provider, Encryptor, Models)."""
    # 1. Database Connection (PostgreSQL or SQLite fallback for serverless cloud)
    db_mode = "postgres"
    try:
        engine = get_engine()
        init_db(engine)
    except Exception:
        db_mode = "sqlite"
        from sqlalchemy import create_engine

        engine = create_engine("sqlite:///finvault_local.db")
        init_db(engine)

    # 2. Master Key and Envelope Encryption (AES-256-GCM)
    key_path = Path(".secrets/master.key")
    key_provider = LocalKeyProvider(key_path)
    encryptor = EnvelopeEncryptor(key_provider)

    # 3. Dense Embedding Provider (Sentence Transformers local)
    embedding_provider = LocalEmbeddingProvider(settings.finvault_embedding_model)

    # 4. Vector Store (Qdrant Remote or In-Memory fallback)
    vector_mode = "qdrant_cloud"
    try:
        vector_store = QdrantStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=embedding_provider.dimension,
        )
    except Exception:
        vector_mode = "in_memory"
        vector_store = QdrantStore(
            url=":memory:",
            collection="finvault_chunks",
            dimension=embedding_provider.dimension,
        )

    # 5. Audit, Session, Review, Graph Stores
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
        "db_mode": db_mode,
        "vector_mode": vector_mode,
        "encryptor": encryptor,
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "retriever": retriever,
        "audit_log": audit_log,
        "session_store": session_store,
        "review_queue": review_queue,
        "graph_store": graph_store,
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
    # Options come from the Role enum itself. Hardcoding a list here let it
    # drift: "executive", "auditor" and "guest" were offered but are not roles
    # the domain model knows, so choosing one was a ValidationError.
    role_options = [r.value for r in Role]
    user_role = st.selectbox(
        "User Role",
        options=role_options,
        index=role_options.index(Role.ANALYST.value),
    )
    org_id = st.text_input("Tenant / Organization ID", value="org_default")

    current_role = Role(user_role)
    # Clearance is DERIVED from role, never chosen alongside it — ROLE_RANK vs
    # CLASSIFICATION_MIN_ROLE in models.py is the only clearance primitive in
    # the system. A separate picker implied the two could disagree, and since
    # User has no `clearance` field, whatever it was set to was silently
    # dropped by pydantic while the demo's gating still read it.
    user_clearance = clearance_ceiling(current_role).value
    st.text_input(
        "Clearance Level (derived from role)",
        value=user_clearance,
        disabled=True,
        help=(
            "Not an independent setting: the highest classification this role may "
            "retrieve, computed from ROLE_RANK and CLASSIFICATION_MIN_ROLE."
        ),
    )

    # Construct the User object
    current_user = User(
        id=user_id,
        username=user_id,
        role=current_role,
        org_id=org_id,
    )

    st.markdown("---")
    st.subheader("⚙️ Model & API Credentials")

    has_live_key = bool(settings.effective_api_key and settings.effective_api_key != "unconfigured")

    provider = st.selectbox(
        "LLM Provider / Route",
        options=["OpenRouter (Default)", "OpenAI", "Moonshot / Kimi", "Custom Endpoint"],
        index=0,
    )

    custom_key = st.text_input(
        "API Key (OpenRouter / OpenAI / Kimi)",
        type="password",
        value=settings.effective_api_key if has_live_key else "",
        placeholder="sk-or-v1-..." if "OpenRouter" in provider else "sk-...",
        help="Enter your API Key. It will be stored in session memory only.",
    )

    model_name = st.text_input(
        "Model Name",
        value=settings.finvault_model,
        help="Default OpenRouter model with tool support: minimax/minimax-m2.7:free or openai/gpt-4o-mini",
    )

    if st.button("💾 Apply API Credentials"):
        if custom_key:
            if "OpenRouter" in provider:
                settings.openrouter_api_key = custom_key
                os.environ["OPENROUTER_API_KEY"] = custom_key
            elif "Moonshot" in provider:
                settings.kimi_api_key = custom_key
                os.environ["KIMI_API_KEY"] = custom_key
            else:
                settings.llm_api_key = custom_key
                os.environ["LLM_API_KEY"] = custom_key
        settings.finvault_model = model_name
        st.success("Credentials updated successfully!")
        st.rerun()

    # Re-check key status
    has_live_key = bool(settings.effective_api_key and settings.effective_api_key != "unconfigured")
    if has_live_key:
        st.markdown('<span class="badge badge-green">🟢 Live LLM Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="badge badge-amber">🟡 Unconfigured Key (Demo Simulation Mode Available)</span>',
            unsafe_allow_html=True,
        )

    demo_mode = st.toggle(
        "⚡ Enable Interactive Demo Mode",
        value=not has_live_key,
        help="Simulate realistic multi-agent execution pipeline without live API keys.",
    )

    with st.expander("ℹ️ How to set Streamlit Secrets"):
        st.markdown(
            """
            In **Streamlit Community Cloud** Settings -> **Secrets**:
            ```toml
            OPENROUTER_API_KEY = "sk-or-v1-..."
            FINVAULT_MODEL = "minimax/minimax-m2.7:free"
            ```
            Or for OpenAI:
            ```toml
            OPENAI_API_KEY = "sk-..."
            FINVAULT_MODEL = "gpt-4o-mini"
            ```
            """
        )

    st.markdown("---")
    st.subheader("🛡️ Security Controls")
    st.markdown(
        f"""
        - <span class="badge badge-green">AES-256 GCM</span> **Active**
        - <span class="badge badge-blue">Prompt Guard</span> **Active**
        - <span class="badge badge-purple">Compliance Veto</span> **Enforced**
        - <span class="badge badge-amber">Token Budget</span> **40k Max**
        - <span class="badge badge-blue">DB Storage</span> **{backend["db_mode"].upper()}**
        - <span class="badge badge-green">Vector Engine</span> **{backend["vector_mode"].upper()}**
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
            {"<span class='badge badge-purple'>Demo Mode: ON</span>" if demo_mode else "<span class='badge badge-green'>Live LLM</span>"}
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
    st.caption(
        "Every query is analyzed by Orchestrator -> Retriever -> Financial Analyst and checked by Compliance Agent."
    )

    # Quick prompt presets
    st.markdown("##### ⚡ Quick Prompts:")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_query = None
    with qcol1:
        if st.button("📊 Q4 Cloud Revenue & Financial Margins"):
            quick_query = "What was the Q4 total revenue and cloud infrastructure revenue growth?"
    with qcol2:
        if st.button("⚖️ AML Suspicious Transaction Threshold"):
            quick_query = "What is the suspicious transaction reporting threshold under the AML compliance policy?"
    with qcol3:
        if st.button("🚫 Restricted Executive Compensation (Test Veto)"):
            quick_query = "What are the executive retention bonuses and confidential compensation figures?"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])
            if msg.get("execution_steps"):
                with st.expander("🔬 View Multi-Agent Execution Steps & Citations", expanded=False):
                    for step in msg["execution_steps"]:
                        status_class = (
                            "step-card-success"
                            if step.get("status") == "success"
                            else ("step-card-error" if step.get("status") == "error" else "step-card-warning")
                        )
                        st.markdown(
                            f"""
                            <div class="step-card {status_class}">
                                <strong>Agent:</strong> <code>{step.get("agent_name", "agent")}</code> | 
                                <strong>Action:</strong> <code>{step.get("name", "step")}</code><br/>
                                <span style="color: #94A3B8; font-size: 0.8rem;">{step.get("output_preview", "")}</span>
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
                                    📄 <strong>Document:</strong> {c.get("document_id", "doc")} | 
                                    <strong>Chunk:</strong> #{c.get("chunk_index", 0)}<br/>
                                    <em>"{c.get("quote", "")}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat Input
    user_prompt = (
        st.chat_input("Ask a question about financial reports, AML/KYC policies, quarterly numbers...") or quick_query
    )
    if user_prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("🤖 Multi-Agent Orchestration in progress (Retriever -> Analyst -> Compliance)..."):
                # Check for Demo Mode vs Live Execution
                if demo_mode or not has_live_key:
                    # Simulated execution pipeline
                    time.sleep(0.8)

                    # 1. Retrieval step from local vector store
                    hits = backend["retriever"].retrieve(
                        query=user_prompt,
                        org_id=org_id,
                        user=current_user,
                        top_k=4,
                    )

                    # Check clearance restrictions. check_clearance is the same
                    # call the retrieval path makes, so the demo cannot claim a
                    # verdict the real access-control module would not reach.
                    may_see_restricted = check_clearance(current_role, Classification.RESTRICTED)
                    is_restricted_query = (
                        "executive" in user_prompt.lower()
                        or "compensation" in user_prompt.lower()
                        or "salary" in user_prompt.lower()
                    )

                    steps_data = [
                        {
                            "name": "orchestrator_planning",
                            "agent_name": "orchestrator",
                            "status": "success",
                            "output_preview": f"Analyzed prompt '{user_prompt[:50]}...'; formulated semantic search plan with Tenant={org_id} and Clearance={user_clearance}.",
                        },
                        {
                            "name": "vector_retrieval",
                            "agent_name": "retriever",
                            "status": "success",
                            "output_preview": f"Retrieved {len(hits)} encrypted chunks; performed AES-256 DEK decryption and relevance scoring.",
                        },
                        {
                            "name": "grounded_analysis",
                            "agent_name": "financial_analyst",
                            "status": "success",
                            "output_preview": "Synthesized financial analysis with grounded citation anchoring.",
                        },
                        {
                            "name": "compliance_guardrail_verification",
                            "agent_name": "compliance_agent",
                            "status": "warning" if (is_restricted_query and not may_see_restricted) else "success",
                            "output_preview": "Enforced externalization policy, PII pattern redactions, and strict grounded claim verification.",
                        },
                    ]

                    if is_restricted_query and not may_see_restricted:
                        res_blocked = True
                        res_reason = "clearance_ceiling_exceeded"
                        res_answer = (
                            f"🛑 **Access Denied / Compliance Veto**: The requested information carries `restricted` classification, "
                            f"which exceeds your current clearance ceiling (`{user_clearance}`). "
                            "This incident has been securely recorded to the tamper-evident audit log."
                        )
                        citations_data = []
                    else:
                        res_blocked = False
                        res_reason = None
                        if hits:
                            doc_sample = hits[0]
                            res_answer = (
                                f"Based on verified corporate filings, here is the grounded summary for your query:\n\n"
                                f"- **Key Metrics**: Verified financial statements indicate strong quarterly execution.\n"
                                f"- **Compliance & Controls**: Transactions meet regulatory reporting thresholds.\n"
                                f"- **Source Grounding**: Derived directly from decrypted document `{doc_sample.document_id}`.\n\n"
                                f"*(Running in interactive multi-agent demo mode. Connect an API Key in the sidebar for live LLM generation)*"
                            )
                            citations_data = [
                                {
                                    "document_id": h.document_id,
                                    "chunk_index": h.chunk_index,
                                    "quote": h.text[:120] + "...",
                                    "classification": h.classification,
                                }
                                for h in hits[:2]
                            ]
                        else:
                            res_answer = (
                                f"No indexed documents matched '{user_prompt}' for tenant `{org_id}`. "
                                "Please upload or ingest sample documents in the **Document Ingestion Vault** tab to test retrieval!"
                            )
                            citations_data = []

                    # Log to audit log
                    backend["audit_log"].append(
                        actor=user_id,
                        action="query_demo",
                        resource="orchestrator",
                        details={"question": user_prompt, "blocked": res_blocked},
                    )

                    if res_blocked:
                        st.error(res_answer)
                    else:
                        st.markdown(res_answer)

                    # Show execution steps
                    with st.expander("🔬 View Agent Execution Trace & Citations", expanded=True):
                        for step in steps_data:
                            status_class = "step-card-success" if step["status"] == "success" else "step-card-warning"
                            st.markdown(
                                f"""
                                <div class="step-card {status_class}">
                                    <strong>Agent:</strong> <code>{step["agent_name"]}</code> | 
                                    <strong>Step:</strong> <code>{step["name"]}</code><br/>
                                    <span style="color: #CBD5E1; font-size: 0.82rem;">{step["output_preview"]}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        if citations_data:
                            st.markdown("##### 📚 Grounded Citations:")
                            for c in citations_data:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        📄 <strong>Document ID:</strong> {c["document_id"]} | 
                                        <strong>Chunk #{c["chunk_index"]}</strong> | 
                                        <strong>Classification:</strong> <span class="badge badge-blue">{c.get("classification", "confidential")}</span><br/>
                                        <span style="color: #94A3B8;">Quote: <em>"{c["quote"]}"</em></span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res_answer,
                            "execution_steps": steps_data,
                            "citations": citations_data,
                        }
                    )

                else:
                    # Live LLM Multi-Agent Orchestration
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

                    try:
                        res: OrchestratorResult = orchestrator.handle(
                            user_prompt,
                            session_id=st.session_state.session_id,
                        )

                        if res.blocked:
                            st.error(f"🛑 **Query Blocked by Compliance Guardrails:** {res.block_reason}")
                            st.warning(res.answer)
                            response_content = f"🛑 **[BLOCKED - {res.block_reason}]**\n\n{res.answer}"
                        else:
                            st.markdown(res.answer)
                            response_content = res.answer

                        steps_data = []
                        if res.execution_steps:
                            with st.expander("🔬 View Agent Execution Trace & Citations", expanded=True):
                                for step in res.execution_steps:
                                    step_dict = {
                                        "name": step.name,
                                        "agent_name": step.agent_name,
                                        "status": step.status.value
                                        if hasattr(step.status, "value")
                                        else str(step.status),
                                        "output_preview": step.output_preview,
                                    }
                                    steps_data.append(step_dict)
                                    status_class = (
                                        "step-card-success" if step_dict["status"] == "success" else "step-card-warning"
                                    )
                                    st.markdown(
                                        f"""
                                        <div class="step-card {status_class}">
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

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_content,
                                "execution_steps": steps_data,
                                "citations": [
                                    {"document_id": c.document_id, "chunk_index": c.chunk_index, "quote": c.quote}
                                    for c in res.citations
                                ],
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Execution Error: {e!s}")


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
        st.markdown("#### 📦 Built-in Sample Financial Documents")
        st.markdown("Quickly test ingestion with built-in sample enterprise files:")
        sample_docs_dir = ROOT_DIR / "sample_docs"
        if sample_docs_dir.exists():
            sample_files = list(sample_docs_dir.glob("*.txt")) + list(sample_docs_dir.glob("*.csv"))
            for sf in sample_files:
                col_btn, col_info = st.columns([1.5, 1])
                with col_btn:
                    if st.button(f"📥 Ingest: {sf.name}", key=f"sample_{sf.name}"):
                        with st.spinner(f"Ingesting {sf.name}..."):
                            content = sf.read_bytes()
                            tier = (
                                "restricted"
                                if "restricted" in sf.name.lower()
                                else ("internal" if "quarterly" in sf.name.lower() else "confidential")
                            )
                            res = backend["ingestion_pipeline"].ingest_file(
                                filename=sf.name,
                                content_bytes=content,
                                org_id=org_id,
                                actor_id=user_id,
                                classification=tier,
                                enable_extraction=enable_graph,
                            )
                            st.success(f"Ingested `{sf.name}` ({res.chunk_count} chunks, {tier})")
                with col_info:
                    st.caption(f"Size: {len(sf.read_bytes())} bytes")

    st.markdown("---")
    st.markdown("#### 📚 Ingested Documents in Vault")
    try:
        from sqlalchemy import text

        with backend["engine"].connect() as conn:
            docs = conn.execute(
                text(
                    "SELECT id, title, classification, org_id, created_at FROM documents WHERE org_id = :org ORDER BY created_at DESC LIMIT 20"
                ),
                {"org": org_id},
            ).fetchall()
            if docs:
                st.table(
                    [
                        {
                            "Doc ID": r[0][:12] + "...",
                            "Title": r[1],
                            "Classification": r[2],
                            "Tenant Org": r[3],
                            "Ingested At": str(r[4])[:19],
                        }
                        for r in docs
                    ]
                )
            else:
                st.info(
                    "No documents ingested yet for this tenant org. Click one of the sample buttons above to populate!"
                )
    except Exception as e:
        st.info(f"Document store status: {e}")


# ==========================================
# TAB 3: AUDIT & COMPLIANCE
# ==========================================
with tab_compliance:
    st.subheader("🔍 Tamper-Evident Audit Logs & Compliance Review")
    st.caption(
        "Every retrieval, prompt externalization, decryption, and agent decision is logged with SHA-256 hash chaining."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Active Encryption", value="AES-256-GCM (DEK/KEK)")
    with col_b:
        st.metric(label="Compliance Policy", value="Strict Grounding Veto")
    with col_c:
        st.metric(label="Tamper Detection", value="SHA-256 Hash Chain")

    st.markdown("---")
    st.markdown("#### 📋 Tamper-Evident Audit Trail")
    try:
        from sqlalchemy import text

        with backend["engine"].connect() as conn:
            logs = conn.execute(
                text(
                    "SELECT seq, timestamp, actor, action, resource, details, entry_hash FROM audit_log ORDER BY seq DESC LIMIT 25"
                )
            ).fetchall()
            if logs:
                st.table(
                    [
                        {
                            "Seq": r[0],
                            "Time": time.strftime("%H:%M:%S", time.localtime(r[1])),
                            "Actor": r[2],
                            "Action": r[3],
                            "Resource": r[4],
                            "Details": str(r[5])[:60] + "...",
                            "Hash": r[6][:12] + "...",
                        }
                        for r in logs
                    ]
                )
            else:
                st.info("No audit logs recorded yet in this session.")
    except Exception as e:
        st.info(f"Audit log query: {e}")

    st.markdown("---")
    st.markdown("#### 🛑 Compliance Review Queue (Blocked Responses)")
    try:
        with backend["engine"].connect() as conn:
            reviews = conn.execute(
                text(
                    "SELECT id, user_id, question, block_reason, status, created_at FROM review_queue WHERE org_id = :org ORDER BY created_at DESC LIMIT 10"
                ),
                {"org": org_id},
            ).fetchall()
            if reviews:
                for rev in reviews:
                    st.warning(
                        f"**Item ID:** `{rev[0]}` | **User:** `{rev[1]}` | **Status:** `{rev[4]}`\n\n"
                        f"**Question:** {rev[2]}\n\n"
                        f"**Block Reason:** `{rev[3]}`"
                    )
            else:
                st.success("✅ Review queue is clean — no pending compliance policy blocks.")
    except Exception as e:
        st.info(f"Review queue status: {e}")


# ==========================================
# TAB 4: ARCHITECTURE & ZERO-TRUST
# ==========================================
with tab_architecture:
    st.subheader("📐 System Architecture & Zero-Trust Threat Model")
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
        ### 🛡️ Five Invariant Security Steps:
        
        1. **Ingestion & Envelope Encryption (DEK/KEK)**:
           - Each document chunk is encrypted using a unique AES-256-GCM Data Encryption Key (DEK).
           - DEKs are wrapped with a Master Key-Encryption-Key (KEK).
           - Only ciphertext is stored in the vector database; vector store compromise yields zero plaintext.
        
        2. **Multi-Tenancy & Tenant Isolation Filter**:
           - All retrieval queries include strict `org_id` vector filtering and PostgreSQL Row-Level Security (RLS).
           - Prevents cross-tenant data contamination across enterprise divisions.
        
        3. **Clearance Level Ceiling Barrier**:
           - The Retriever filters out any documents whose classification tier (`confidential`, `restricted`) exceeds the user's active clearance.
           - Restricted content is never passed to external LLM endpoints.
        
        4. **Grounded Financial Analyst & Token Budget**:
           - Analyst Agent outputs structured JSON containing verifiable exact-quote citations.
           - A hard 40,000 token budget ceiling prevents runaway recursive loops and denial-of-wallet attacks.
        
        5. **Compliance Agent Fail-Closed Policy Enforcement**:
           - Runs deterministically as Python control flow (cannot be skipped by the Orchestrator LLM).
           - Scans for PII / API keys and verifies that every cited quote exists in the retrieved context.
           - Vetoes ungrounded claims and routes blocked responses to the quarantine review queue.
           - Hash-chains all events into an append-only audit trail (`entry_hash = SHA256(prev_hash + entry)`).
        """
    )
