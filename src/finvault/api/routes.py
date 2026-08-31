"""API routes: auth, document ingestion, and querying.

Every route that touches data resolves the acting User from a verified JWT
(get_current_user) — never from a client-supplied org_id/role — so
access-control decisions downstream are anchored to something the server
issued, not something the caller asserted.

Error handling in this file follows one rule: **routes raise, they do not
translate.** No route builds an `HTTPException`, chooses a status code, or
writes a client-facing error string. They raise the domain errors from
`finvault/errors.py`, and `api/error_handlers.py` turns those into
responses — so the status code for, say, "no clearance" is defined once,
not once per route.

There are two deliberate exceptions, both on the query path, and both
because the endpoint's contract is to return a *result* rather than fail:
POST /query and POST /query/stream catch everything and return a blocked
QueryResponse. See `_unavailable_response` below.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finvault.agents.base import TokenBudget
from finvault.agents.canvas_models import ComparisonHeatmap, ExecutionStepNode, KnowledgeGraphData
from finvault.agents.comparison_agent import ComparisonAgent
from finvault.agents.compliance_agent import ComplianceAgent
from finvault.agents.execution_events import ExecutionEvent, ExecutionEventBus
from finvault.agents.orchestrator import Orchestrator, OrchestratorResult
from finvault.api.auth import create_access_token, get_current_user
from finvault.api.uploads import spooled_upload
from finvault.cache import Cache, bump_corpus_generation, corpus_generation, scoped_key
from finvault.config import settings
from finvault.errors import (
    AccessDeniedError,
    AgentExecutionError,
    FinVaultError,
    InvalidRequestError,
    NotFoundError,
)
from finvault.ingestion.loaders import load_text
from finvault.metrics import (
    documents_ingested_total,
    ingest_duration_seconds,
    observe_duration,
    record_cache,
)
from finvault.metrics import (
    render as render_metrics,
)
from finvault.models import ROLE_RANK, Classification, Role, User
from finvault.observability import capture_context, extra_fields, get_logger, log_exception, run_in_request_context
from finvault.security.access_control import require_clearance
from finvault.security.review_queue import ReviewItem

logger = get_logger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    role: Role
    org_id: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/token", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    """Dev/demo login: issues a token for any username+role+org_id claimed by
    the caller. This is deliberately not real authentication — replace with
    credential verification against your identity provider (SSO, password
    hash, etc.) before any non-local deployment. It exists so the rest of the
    API — which *does* enforce real access control once a token is issued —
    is testable end-to-end without standing up a full IdP.
    """
    user = User(username=payload.username, role=payload.role, org_id=payload.org_id)
    token = create_access_token(user=user)
    return LoginResponse(access_token=token)


class QueryRequest(BaseModel):
    question: str
    # Omit to start a new session; pass back the session_id a prior response
    # returned to continue the conversation with history (see agents/session.py).
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    blocked: bool
    block_reason: str | None
    guardrail_findings: list[str]
    injection_flags_detected: int
    citations: list[dict]
    session_id: str
    # Populated by OrchestratorResult — execution_steps from the event bus
    # (see agents/execution_events.py), graph_data from the graph retriever
    # scoped to this query's retrieved documents (see agents/orchestrator.py).
    execution_steps: list[ExecutionStepNode] = Field(default_factory=list)
    graph_data: KnowledgeGraphData = Field(default_factory=KnowledgeGraphData)


def _build_query_response(result: OrchestratorResult, *, session_id: str) -> QueryResponse:
    """Single source of truth for OrchestratorResult -> QueryResponse
    shaping — both POST /query and POST /query/stream's terminal "done"
    event go through this, so the two response shapes can't silently drift
    apart the way execution_steps/graph_data already did once (Milestone 0).
    """
    return QueryResponse(
        answer=result.answer,
        blocked=result.blocked,
        block_reason=result.block_reason,
        guardrail_findings=result.guardrail_findings,
        injection_flags_detected=len(result.injection_flags),
        citations=result.citations,
        session_id=session_id,
        execution_steps=result.execution_steps,
        graph_data=result.graph_data,
    )


def _build_orchestrator(request: Request, user: User) -> Orchestrator:
    return Orchestrator(
        retriever=request.app.state.retriever,
        user=user,
        audit_log=request.app.state.audit_log,
        compliance_agent=ComplianceAgent(),
        session_store=request.app.state.session_store,
        review_queue=request.app.state.review_queue,
        graph_retriever=request.app.state.graph_retriever,
    )


def _unavailable_response(exc: BaseException, *, session_id: str) -> QueryResponse:
    """The fail-closed shape both query endpoints return when the pipeline
    raises something Orchestrator.handle() did not already convert into a
    blocked result.

    `block_reason` is the exception's `code` from `errors.py` when it has
    one, so the reason a query failed is drawn from the same vocabulary as
    every API error and every log line — the frontend and the logs agree on
    the string, instead of this route inventing "internal_error" on its own.
    Anything that is not a FinVaultError is a bug, and reports as one.
    """
    code = exc.code if isinstance(exc, FinVaultError) else "internal_error"
    message = exc.user_message if isinstance(exc, FinVaultError) else _UNAVAILABLE_ANSWER
    return QueryResponse(
        answer=message,
        blocked=True,
        block_reason=code,
        guardrail_findings=[],
        injection_flags_detected=0,
        citations=[],
        session_id=session_id,
    )


_UNAVAILABLE_ANSWER = (
    "The assistant is temporarily unavailable and could not process this request. Please try again shortly."
)


def _query_cache_key(question: str, *, user: User, cache: Cache) -> str:
    """The cache key for one answer.

    Four things go into it, and leaving out any one of them is a bug with a
    security consequence rather than a performance one:

    - **org_id** — the obvious tenant boundary.
    - **role** — two users in the same org are entitled to different answers
      to the identical question, because clearance decides what retrieval
      returned. A key without the role serves whichever answer was computed
      first, which is a clearance bypass whenever the first asker outranked
      the second.
    - **the question**, normalized only for surrounding whitespace. Nothing
      more aggressive: case and punctuation change what an embedding
      retrieves, so folding them would serve an answer to a different query.
    - **the corpus generation**, so a document ingested a second ago retires
      every answer cached before it (see cache.py). TTL alone cannot do
      this — a user who uploads a policy and immediately asks about it must
      not be told what was true before the upload.

    `scoped_key` builds the digest, so org and role cannot be omitted by
    accident at this call site or any future one.
    """
    generation = corpus_generation(cache, user.org_id)
    return scoped_key(
        "answer",
        question.strip(),
        str(generation),
        org_id=user.org_id,
        role=user.role.value,
    )


def _cacheable(payload: QueryRequest, result: OrchestratorResult) -> bool:
    """Whether this answer may be stored.

    Three exclusions, each for a distinct reason:

    - **A session turn** depends on conversation history, so the same
      question in two conversations has two correct answers. The key does
      not encode history and deliberately will not — caching a follow-up
      like "and the year before?" would be answering it from someone else's
      conversation.
    - **A blocked result** is a compliance decision about one request. It is
      cheap to recompute and must be re-decided, not replayed.
    - **An answer with no citations** came from a degraded path (the model
      answered without calling analyze — see orchestrator.py). Caching a
      degraded answer makes one bad turn durable.
    """
    return payload.session_id is None and not result.blocked and bool(result.citations)


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request, user: User = Depends(get_current_user)) -> QueryResponse:
    """Answers a question against the corpus the caller is cleared to read.

    Never raises. Orchestrator.handle() already converts an agent failure
    into a blocked OrchestratorResult (see agents/orchestrator.py); the
    catch below is the backstop for everything it does not — a store that
    is down, a bug in response shaping. Returning HTTP 200 with
    `blocked=true` rather than a 5xx is deliberate: a partial trace and a
    stated reason are more useful to the frontend than an error page, and
    "no answer" is already a first-class outcome of this endpoint.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    cache: Cache = request.app.state.cache

    if settings.finvault_enable_query_cache and payload.session_id is None:
        cached = cache.get(_query_cache_key(payload.question, user=user, cache=cache))
        if cached is not None:
            record_cache("query", result="hit")
            logger.info("query_cache_hit", extra=extra_fields(session_id=session_id))
            # The audit trail records the answer being served, not the work
            # that produced it — a cached answer is still a disclosure of
            # document-derived content to this user at this moment, and an
            # audit log that only saw uncached queries would under-record
            # exactly the repeated access patterns an auditor looks for.
            request.app.state.audit_log.append(
                actor=user.id, action="query_cached", resource="orchestrator", details={"question": payload.question}
            )
            return QueryResponse(**{**cached, "session_id": session_id})
        record_cache("query", result="miss")

    orchestrator = _build_orchestrator(request, user)
    try:
        result = orchestrator.handle(payload.question, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 — last line of defense: this route must never 500
        log_exception(logger, exc, "query_failed", session_id=session_id)
        request.app.state.audit_log.append(
            actor=user.id, action="agent_failure", resource="orchestrator", details={"error": f"unhandled: {exc}"}
        )
        return _unavailable_response(exc, session_id=session_id)

    response = _build_query_response(result, session_id=session_id)
    if settings.finvault_enable_query_cache and _cacheable(payload, result):
        stored = response.model_dump(mode="json")
        # The execution trace is dropped before storing. It is a record of
        # what *this* request actually did — real per-step durations from a
        # real run — and replaying it against a later request would present
        # fabricated timings as measurements, which is precisely what
        # orchestrator.py's _run_step exists to avoid.
        stored.pop("execution_steps", None)
        cache.set(
            _query_cache_key(payload.question, user=user, cache=cache),
            stored,
            ttl_seconds=settings.finvault_query_cache_ttl_seconds,
        )
        record_cache("query", result="store")
    return response


_QUEUE_DONE = object()


def _sse_message(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/query/stream")
async def query_stream(
    payload: QueryRequest, request: Request, user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Live execution trace as a query runs — same pipeline as POST /query,
    streamed as Server-Sent Events instead of returned all at once.

    Not native EventSource on the frontend: EventSource can't send the
    Authorization header this app's auth model requires, so the frontend
    uses fetch() + a manual ReadableStream reader instead (see
    frontend/index.html) — same SSE wire format, just not the browser's
    built-in client, which is also why this is POST (EventSource is
    GET-only) with the same JSON body shape as /query, rather than the
    question text sitting in a URL.

    Orchestrator.handle() is synchronous and can take real wall-clock time
    (several LLM calls) before returning, so it runs in a background
    thread; events cross into this async generator through a thread-safe
    queue.Queue, drained via run_in_executor so the event loop isn't
    blocked waiting on it.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    orchestrator = _build_orchestrator(request, user)

    event_bus = ExecutionEventBus()
    event_queue: queue.Queue = queue.Queue()
    event_bus.subscribe(event_queue.put)

    # A raw Thread does not inherit contextvars the way asyncio.to_thread
    # does, so the whole context is captured here and restored inside the
    # thread. Two of the request's contextvars matter and both are load-
    # bearing: the request id, without which every log line from the
    # orchestrator running under /query/stream would be missing the id that
    # ties it to the request — precisely when correlation matters most —
    # and the tenant scope from security/rls.py, without which every
    # database transaction in the thread runs with an empty
    # `app.current_org`: session history and graph data come back empty and
    # the session/review-queue writes fail the policy's WITH CHECK.
    context = capture_context()

    def run_orchestrator() -> None:
        def _work() -> None:
            try:
                result = orchestrator.handle(payload.question, session_id=session_id, event_bus=event_bus)
                event_queue.put(("done", result))
            except Exception as exc:  # noqa: BLE001 — last line of defense, mirrors POST /query's own backstop
                log_exception(logger, exc, "query_stream_failed", session_id=session_id)
                request.app.state.audit_log.append(
                    actor=user.id,
                    action="agent_failure",
                    resource="orchestrator",
                    details={"error": f"unhandled: {exc}"},
                )
                event_queue.put(("error", exc))
            finally:
                event_queue.put(_QUEUE_DONE)

        run_in_request_context(context, _work)

    threading.Thread(target=run_orchestrator, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, event_queue.get)
            if item is _QUEUE_DONE:
                break

            if isinstance(item, tuple):
                kind, value = item
                if kind == "done":
                    response = _build_query_response(value, session_id=session_id)
                    yield _sse_message("done", response.model_dump())
                else:  # "error" — `value` is the exception the worker caught
                    response = _unavailable_response(value, session_id=session_id)
                    yield _sse_message("error", response.model_dump())
                continue

            event: ExecutionEvent = item
            if event.type == "step_started":
                yield _sse_message("step_started", {"agent": event.agent, "action": event.action})
            else:  # "step_finished"
                yield _sse_message(
                    "step_finished", {"agent": event.agent, "action": event.action, "step": event.step.model_dump()}
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class IngestResponse(BaseModel):
    document_id: str
    title: str
    classification: Classification


@router.post("/documents", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile,
    classification: Classification,
    user: User = Depends(get_current_user),
) -> IngestResponse:
    # Only roles cleared to *read* a classification may ingest documents at
    # that classification — otherwise a low-clearance user could write
    # content into the corpus at a tier they couldn't themselves read back,
    # bypassing the access-control model from the write side.
    #
    # No try/except: require_clearance raises AccessDeniedError, which is
    # already a 403 with safe wording (see errors.py). Catching it here only
    # to re-raise it as an HTTPException is the duplication this redesign
    # removed.
    require_clearance(user.role, classification, resource="document")

    # Size enforcement and temp-file cleanup both live in spooled_upload —
    # including on the failure paths, which is where the previous inline
    # version was easiest to get wrong. UnsupportedDocumentError (415) from
    # the loader and PayloadTooLargeError (413) from here both propagate
    # untouched to api/error_handlers.py.
    # Two separate statements, not one `async with A(), B()`: every manager
    # in a single `async with` must be an async one, and observe_duration is
    # a plain @contextmanager. Combining them raises TypeError before the
    # body ever runs — which made this endpoint a guaranteed 500.
    async with spooled_upload(file) as tmp_path:
        with observe_duration(ingest_duration_seconds):
            pipeline = request.app.state.ingestion_pipeline
            # ingest_file is synchronous and can run for a long time (embedding
            # every chunk, plus a real LLM call for entity extraction) — run it
            # in a worker thread so it can't block this process's single event
            # loop from servicing anything else (health checks, other requests,
            # SSE streams already in flight) for the duration of one ingest.
            document = await asyncio.to_thread(
                pipeline.ingest_file,
                tmp_path,
                org_id=user.org_id,
                classification=classification,
                title=file.filename,
                actor=user.id,
            )

    # Retires every cached answer for this org. Without it, a user who
    # uploads a document and immediately asks about it would be served an
    # answer computed before the upload — the single most confusing possible
    # behavior for a document system, and the reason the cache key carries a
    # generation counter at all (see cache.py).
    bump_corpus_generation(request.app.state.cache, user.org_id)

    documents_ingested_total.labels(classification=classification.value).inc()
    logger.info(
        "document_ingested",
        extra=extra_fields(document_id=document.id, classification=classification.value),
    )
    return IngestResponse(document_id=document.id, title=document.title, classification=document.classification)


class ClassificationSuggestionResponse(BaseModel):
    suggested_classification: Classification
    confidence: float
    scores: dict[str, float]


@router.post("/documents/classification-suggestion", response_model=ClassificationSuggestionResponse)
async def suggest_document_classification(
    request: Request,
    file: UploadFile,
    user: User = Depends(get_current_user),
) -> ClassificationSuggestionResponse:
    """Advisory preview only: does not ingest anything or persist any state.
    Lets a caller (e.g. the upload UI) show a suggested tier before the user
    commits to one on the actual POST /documents call. The suggester never
    gets the final say — see ingestion/classification.py.

    Same size cap and thread-offloading as POST /documents — this endpoint
    reads and classifies the file just as synchronously, so without both, an
    oversized file here would hang the event loop (no size limit at all) or
    block it (no offload) exactly like the ingest route did before this fix.
    Both now come from the one spooled_upload helper the ingest route uses,
    so the two can no longer enforce different limits by accident.
    """
    async with spooled_upload(file) as tmp_path:
        text = await asyncio.to_thread(load_text, tmp_path)

    suggestion = await asyncio.to_thread(request.app.state.classification_suggester.suggest, text)
    return ClassificationSuggestionResponse(
        suggested_classification=suggestion.predicted,
        confidence=suggestion.confidence,
        scores=suggestion.scores,
    )


class CompareRequest(BaseModel):
    document_ids: list[str]


@router.post("/documents/compare", response_model=ComparisonHeatmap)
def compare_documents(
    payload: CompareRequest, request: Request, user: User = Depends(get_current_user)
) -> ComparisonHeatmap:
    """Dynamic Multi-Document Difference & Risk Heatmap: reassembles each
    requested document's full plaintext (ACL-gated per document, silently
    dropping any the caller can't access — same non-inference posture as
    /query, see Retriever.get_document_text), then asks ComparisonAgent for
    comparable metrics and a deterministic variance-based risk score per
    metric (see agents/comparison_agent.py — the LLM never sets a risk score
    itself).
    """
    if len(payload.document_ids) < 2:
        raise InvalidRequestError(
            "compare_documents called with fewer than two ids",
            context={"requested": len(payload.document_ids)},
            user_message="At least two document_ids are required for comparison.",
        )

    retriever = request.app.state.retriever
    documents: list[tuple[str, str]] = []
    for document_id in payload.document_ids:
        retrieved = retriever.get_document_text(document_id, user=user)
        if retrieved is not None:
            documents.append((retrieved.title, retrieved.text))

    if len(documents) < 2:
        # 404 rather than 403 even when the cause is clearance: saying
        # "forbidden" would confirm the document exists, which the retrieval
        # layer deliberately refuses to do (see Retriever.get_document_text).
        raise NotFoundError(
            "Fewer than two requested documents were accessible",
            context={"requested": len(payload.document_ids), "accessible": len(documents)},
            user_message="Fewer than two of the requested documents were found and accessible to you.",
        )

    comparison_agent = ComparisonAgent()
    try:
        heatmap = comparison_agent.compare(
            documents, budget=TokenBudget(limit=settings.finvault_max_tokens_per_request)
        )
    except AgentExecutionError as exc:
        # Audit-logged *and* re-raised: the audit trail records that a
        # compliance-relevant operation failed, while the exception carries
        # the 503 and the safe wording to the client. Unlike /query, this
        # endpoint has no partial result worth returning, so it fails loudly.
        log_exception(logger, exc, "comparison_failed", document_count=len(documents))
        request.app.state.audit_log.append(
            actor=user.id, action="agent_failure", resource="comparison", details={"error": str(exc)}
        )
        raise

    request.app.state.audit_log.append(
        actor=user.id,
        action="compare_documents",
        resource=",".join(payload.document_ids),
        details={
            "requested": len(payload.document_ids),
            "compared": len(documents),
            "metrics_found": len(heatmap.metrics),
            "injection_flags": comparison_agent.last_injection_flags,
        },
    )
    return heatmap


def _require_compliance_role(user: User) -> None:
    """Guards the compliance surface. Raises AccessDeniedError (403) — the
    same error type every other clearance check in the system raises, so
    "denied" looks identical in the logs whether it came from RBAC here or
    from classification clearance in security/access_control.py.
    """
    if ROLE_RANK[user.role] < ROLE_RANK[Role.COMPLIANCE_OFFICER]:
        raise AccessDeniedError(
            f"Role '{user.role.value}' is not authorized for the compliance review queue",
            context={"role": user.role.value, "required_role": Role.COMPLIANCE_OFFICER.value},
            user_message="You are not authorized to access the compliance review queue.",
        )


class ReviewItemResponse(BaseModel):
    id: str
    question: str
    draft_answer: str
    block_reason: str | None
    findings: list[str]
    citations: list[dict]
    status: str
    created_at: float
    reviewed_by: str | None
    reviewed_at: float | None
    reviewer_note: str | None


def _to_review_item_response(item: ReviewItem) -> ReviewItemResponse:
    return ReviewItemResponse(
        id=item.id,
        question=item.question,
        draft_answer=item.draft_answer,
        block_reason=item.block_reason,
        findings=item.findings,
        citations=item.citations,
        status=item.status,
        created_at=item.created_at,
        reviewed_by=item.reviewed_by,
        reviewed_at=item.reviewed_at,
        reviewer_note=item.reviewer_note,
    )


@router.get("/compliance/review-queue", response_model=list[ReviewItemResponse])
def list_review_queue(request: Request, user: User = Depends(get_current_user)) -> list[ReviewItemResponse]:
    """Blocked responses awaiting manual handling — see security/review_queue.py.
    Org-scoped: a compliance officer only ever sees their own org's items.
    """
    _require_compliance_role(user)
    items = request.app.state.review_queue.list_pending(org_id=user.org_id)
    return [_to_review_item_response(item) for item in items]


class ResolveReviewRequest(BaseModel):
    status: Literal["released", "denied"]
    note: str | None = None


@router.post("/compliance/review-queue/{item_id}/resolve", response_model=ReviewItemResponse)
def resolve_review_item(
    item_id: str, payload: ResolveReviewRequest, request: Request, user: User = Depends(get_current_user)
) -> ReviewItemResponse:
    """Records a compliance officer's decision on a blocked response. This
    does not re-deliver the answer to the original user automatically — the
    officer's own out-of-band channel (email, ticket, etc.) handles that,
    consistent with the block message's "requires manual handling."
    """
    _require_compliance_role(user)
    # No try/except: the queue raises ReviewItemNotFoundError (404) for an
    # unknown or other-org item and ReviewQueueError (400) for an illegal
    # transition. The previous blanket `except ReviewQueueError -> 404`
    # reported a bad transition as "not found", which is a misleading status
    # for a request whose target exists.
    item = request.app.state.review_queue.resolve(
        item_id, org_id=user.org_id, status=payload.status, reviewed_by=user.id, reviewer_note=payload.note
    )

    request.app.state.audit_log.append(
        actor=user.id,
        action="compliance_review_resolved",
        resource=item_id,
        details={"status": payload.status, "note": payload.note},
    )
    return _to_review_item_response(item)


@router.get("/graph", response_model=KnowledgeGraphData)
def get_graph(request: Request, user: User = Depends(get_current_user)) -> KnowledgeGraphData:
    """The full Knowledge Graph & Lineage Canvas for the caller's org — every
    entity/relationship extracted so far, ACL-filtered (see
    retrieval/graph_retriever.py). Unlike POST /query's graph_data, this is
    not scoped to any one question; it's the standalone canvas view.
    """
    return request.app.state.graph_retriever.get_graph(user=user)


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """Liveness and dependency status.

    `status` stays "ok" while the process can serve requests. The cache is
    reported separately and never degrades it: an unreachable cache makes
    FinVault slower, not broken (see cache.py), and a health check that
    failed on it would have Kubernetes restart healthy pods during a Redis
    blip — turning a cache outage into an application outage.
    """
    cache = getattr(request.app.state, "cache", None)
    return {
        "status": "ok",
        "cache": "up" if cache is not None and cache.available else "degraded",
    }


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus exposition.

    Unauthenticated by design: a scraper holding a bearer token is a
    credential distribution problem, and this endpoint exposes counters and
    latencies — never document content, never identity. Restrict it at the
    network layer instead (a NetworkPolicy, or the Helm chart's ingress,
    which excludes this path). `include_in_schema=False` keeps it out of the
    public OpenAPI document, where it is noise.
    """
    if not settings.finvault_enable_metrics:
        raise NotFoundError("Metrics endpoint is disabled", context={"setting": "finvault_enable_metrics"})
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
