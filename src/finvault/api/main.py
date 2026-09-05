"""FastAPI application wiring.

All infra singletons (key provider, encryptor, embedding model, vector store,
audit log, DB engine) are constructed once at startup and stored on
app.state — every route reads them from there rather than constructing its
own, so there's exactly one KEK, one embedding model load, and one DB
connection pool per process.

Two things happen here that the rest of the app depends on and cannot do
for itself:

- `configure_logging()` runs before anything else, so a failure during
  startup itself is logged in the same format as everything after it.
- `install_error_handlers(app)` registers the request-id middleware and the
  four exception handlers. Without it, routes raising `FinVaultError` would
  surface as bare 500s — the handlers are what give those errors their
  status codes. See `api/error_handlers.py`.

Startup is deliberately fail-fast: a dependency that cannot be constructed
(unreachable Postgres, missing master key) raises out of `lifespan` and the
process exits instead of serving traffic that would fail one request at a
time.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from finvault.agents.session import PostgresSessionStore
from finvault.api.error_handlers import install_error_handlers
from finvault.cache import build_cache
from finvault.config import settings
from finvault.db import get_engine, init_db
from finvault.errors import ConfigurationError
from finvault.ingestion.classification import ClassificationSuggester
from finvault.ingestion.embeddings import CachedEmbeddingProvider, LocalEmbeddingProvider
from finvault.ingestion.extraction import ExtractionAgent
from finvault.ingestion.pipeline import IngestionPipeline
from finvault.observability import configure_logging, extra_fields, get_logger
from finvault.retrieval.graph_retriever import GraphRetriever
from finvault.retrieval.graph_store import PostgresGraphStore
from finvault.retrieval.reranker import LocalCrossEncoderReranker
from finvault.retrieval.retriever import Retriever
from finvault.retrieval.vector_store import QdrantStore
from finvault.security.audit import PostgresAuditLog
from finvault.security.encryption import EnvelopeEncryptor
from finvault.security.kms import build_key_provider
from finvault.security.quarantine import PostgresQuarantineStore
from finvault.security.review_queue import PostgresReviewQueue
from finvault.security.rls import enable_row_level_security, install_org_scoping, verify_isolation

logger = get_logger(__name__)


def _resolve_frontend_dir() -> Path:
    for candidate in [
        Path("/app/frontend"),
        Path.cwd() / "frontend",
        Path(__file__).resolve().parents[3] / "frontend",
    ]:
        if candidate.is_dir():
            return candidate
    return Path("/app/frontend")


# Resolves frontend dir across dev checkout, installed package, and Docker container (/app/frontend)
FRONTEND_DIR = _resolve_frontend_dir()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info(
        "startup_begin",
        extra=extra_fields(model=settings.finvault_model, qdrant_url=settings.qdrant_url),
    )
    engine = get_engine()
    if settings.finvault_auto_create_schema:
        init_db(engine)

    # Tenant isolation the database enforces, not just the queries (see
    # security/rls.py). Installed before anything reads a tenant table:
    # org scoping has to be on the engine from the first transaction, or an
    # early read would run without a policy context and silently see nothing.
    install_org_scoping(engine)
    if settings.finvault_enable_rls:
        enable_row_level_security(engine)
        unprotected = sorted(t for t, ok in verify_isolation(engine).items() if not ok)
        if unprotected:
            # Fail startup rather than serve. Every way RLS can be silently
            # wrong (ENABLE without FORCE, a policy dropped by a migration)
            # looks identical to working from inside the application, and
            # the failure mode is cross-tenant disclosure.
            raise ConfigurationError(
                "Row Level Security is enabled but is not actually enforced. The two usual "
                "causes are connecting as a superuser or BYPASSRLS role (policies do not "
                "apply to it at all), or a table with ENABLE but not FORCE (its owner is "
                "exempt). Check the rls_ineffective_connection_role / rls_verification_failed "
                "log record just above for which one. See security/rls.py.",
                context={"unprotected_tables": unprotected},
            )

    cache = build_cache()
    app.state.cache = cache

    key_provider = build_key_provider()
    encryptor = EnvelopeEncryptor(key_provider)
    embedding_provider: LocalEmbeddingProvider | CachedEmbeddingProvider = LocalEmbeddingProvider(
        settings.finvault_embedding_model
    )
    if settings.finvault_enable_embedding_cache:
        # Wraps rather than replaces, and reports the inner model's name, so
        # a corpus ingested with the cache on stays readable with it off —
        # see CachedEmbeddingProvider for why that matters at query time.
        embedding_provider = CachedEmbeddingProvider(
            embedding_provider, cache, ttl_seconds=settings.finvault_embedding_cache_ttl_seconds
        )
    vector_store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimension=embedding_provider.dimension,
        api_key=settings.qdrant_api_key,
    )
    audit_log = PostgresAuditLog(engine)

    app.state.db_engine = engine
    app.state.encryptor = encryptor
    app.state.embedding_provider = embedding_provider
    app.state.vector_store = vector_store
    app.state.audit_log = audit_log
    app.state.session_store = PostgresSessionStore(engine)
    quarantine_store = PostgresQuarantineStore(engine)
    app.state.quarantine_store = quarantine_store
    app.state.review_queue = PostgresReviewQueue(engine)
    reranker = (
        LocalCrossEncoderReranker(settings.finvault_cross_encoder_model)
        if settings.finvault_enable_cross_encoder_rerank
        else None
    )
    app.state.retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        reranker=reranker,
        quarantine_store=quarantine_store,
    )
    classification_suggester = ClassificationSuggester(embedding_provider)
    app.state.classification_suggester = classification_suggester
    extraction_agent = ExtractionAgent()
    graph_store = PostgresGraphStore(engine)
    app.state.extraction_agent = extraction_agent
    app.state.graph_store = graph_store
    app.state.graph_retriever = GraphRetriever(graph_store=graph_store, encryptor=encryptor)
    app.state.ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        encryptor=encryptor,
        audit_log=audit_log,
        db_engine=engine,
        classification_suggester=classification_suggester,
        extraction_agent=extraction_agent,
        graph_store=graph_store,
    )

    logger.info(
        "startup_complete",
        extra=extra_fields(
            cross_encoder_rerank=reranker is not None,
            cache_backend=type(cache).__name__,
            cache_available=cache.available,
            rls_enabled=settings.finvault_enable_rls,
            rate_limit_enabled=settings.finvault_enable_rate_limit,
        ),
    )
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FinVault",
        description="Secure multi-agent RAG platform for finance.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware and exception handlers before routes: the handlers are what
    # give every FinVaultError its status code and envelope, so a route
    # registered without them would fail in a shape nothing else uses.
    # The cache reaches install_error_handlers because the rate limiter's
    # counters live in it. It is built here rather than read from app.state,
    # which the lifespan has not populated yet at middleware-registration
    # time — middleware is wired once, at import, not per request.
    install_error_handlers(app, cache=build_cache())

    from finvault.api.routes import router

    app.include_router(router)

    # Static frontend, mounted under /app so it never shadows the API routes
    # above (which live at the root path). html=True serves index.html for
    # the mount root and falls back to it for unmatched sub-paths.
    if FRONTEND_DIR.is_dir():
        from fastapi.responses import RedirectResponse

        @app.get("/", include_in_schema=False)
        def index_redirect():
            return RedirectResponse(url="/app")

        app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
