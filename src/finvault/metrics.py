"""Prometheus instrumentation.

Companion to `observability.py`: logs tell you what happened in *one*
request, metrics tell you what is happening across *all* of them. The
division is deliberate — anything with unbounded cardinality (a request id,
a user id, a question) belongs in a log line and must never become a label
here, because Prometheus stores one time series per distinct label
combination and a user-id label on a busy endpoint is how a metrics backend
falls over.

Every label used in this file is drawn from a small closed set: HTTP method,
route template, status class, agent name, error code, cache name.

The four questions these answer, which the logs answer only one request at
a time:

- *Is it up and how fast?*   `finvault_http_request_duration_seconds`
- *What is it costing?*      `finvault_llm_tokens_total`, `finvault_llm_cost_usd_total`
- *Which agent is slow?*     `finvault_agent_duration_seconds`
- *Is the cache earning it?* `finvault_cache_operations_total`

Import is safe without `prometheus_client` installed: the module degrades to
no-op stand-ins so metrics remain an optional deployment concern rather than
a hard dependency of the library.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from finvault.observability import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in installs without the extra
    METRICS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoOpMetric:
        """Accepts every call the real API accepts and does nothing.

        Exists so call sites never guard on availability — an
        `if METRICS_AVAILABLE` around every increment would be noise in the
        business logic, and the one that got forgotten would be an
        AttributeError in production.
        """

        def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None: ...
        def dec(self, *args: Any, **kwargs: Any) -> None: ...
        def set(self, *args: Any, **kwargs: Any) -> None: ...
        def observe(self, *args: Any, **kwargs: Any) -> None: ...

    Counter = Histogram = Gauge = lambda *args, **kwargs: _NoOpMetric()  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]

    def generate_latest(registry: Any = None) -> bytes:  # type: ignore[misc]
        return b"# prometheus_client is not installed\n"


# Buckets tuned to what this system actually does, not the library defaults.
# An LLM-backed query takes seconds, not milliseconds: default buckets top
# out at 10s and would put nearly every /query in +Inf, which is exactly the
# bucket that tells you nothing.
_HTTP_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 45, 90)
_AGENT_BUCKETS = (0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)
_INGEST_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 180, 600)


http_requests_total = Counter(
    "finvault_http_requests_total",
    "HTTP requests by route, method and status class.",
    ["method", "route", "status_class"],
)

http_request_duration_seconds = Histogram(
    "finvault_http_request_duration_seconds",
    "End-to-end HTTP request latency.",
    ["method", "route"],
    buckets=_HTTP_BUCKETS,
)

http_requests_in_flight = Gauge(
    "finvault_http_requests_in_flight",
    "Requests currently being served. A rising floor here means saturation, which latency percentiles alone can hide.",
)

errors_total = Counter(
    "finvault_errors_total",
    "Errors by their code from errors.py and the branch that code belongs to. "
    "Alert on the `incident` branches; the `expected` ones are the system working.",
    ["error_code", "branch"],
)

agent_runs_total = Counter(
    "finvault_agent_runs_total",
    "Agent invocations by name and outcome.",
    ["agent", "outcome"],
)

agent_duration_seconds = Histogram(
    "finvault_agent_duration_seconds",
    "Wall-clock duration of one agent run, including its whole tool-use loop.",
    ["agent"],
    buckets=_AGENT_BUCKETS,
)

llm_requests_total = Counter(
    "finvault_llm_requests_total",
    "Chat-completion requests to the model provider, including retried attempts.",
    ["agent", "model", "outcome"],
)

llm_tokens_total = Counter(
    "finvault_llm_tokens_total",
    "Tokens consumed, split by direction. Multiply by your provider's rate to get spend, "
    "or use finvault_llm_cost_usd_total if rates are configured.",
    ["agent", "model", "direction"],
)

llm_cost_usd_total = Counter(
    "finvault_llm_cost_usd_total",
    "Estimated spend, from the per-million-token rates in settings. Flat zero means the rates "
    "are unconfigured, not that the system is free — see FINVAULT_PRICE_INPUT_PER_MILLION.",
    ["agent", "model"],
)

llm_retries_total = Counter(
    "finvault_llm_retries_total",
    "Retried LLM attempts by the upstream exception type. A rising rate here is the "
    "earliest warning of provider trouble — it precedes user-visible failure.",
    ["agent", "upstream_error"],
)

token_budget_exhausted_total = Counter(
    "finvault_token_budget_exhausted_total",
    "Requests that hit the per-request token ceiling. Sustained non-zero means the budget "
    "is mis-sized or an agent is thrashing.",
    ["agent"],
)

forced_tool_synthesized_total = Counter(
    "finvault_forced_tool_synthesized_total",
    "First-turn tool calls FinVault had to run itself because the model ignored tool_choice. "
    "Sustained non-zero means the configured model does not honor tool forcing — answers are "
    "still grounded, but on a blunter query than the model would have written.",
    ["agent", "tool"],
)

cache_operations_total = Counter(
    "finvault_cache_operations_total",
    "Cache lookups by result. hit/miss give you the hit rate; `unavailable` separates a cold cache from a broken one.",
    ["cache", "result"],
)

rate_limit_decisions_total = Counter(
    "finvault_rate_limit_decisions_total",
    "Rate-limiter verdicts. `failed_open` means the limiter could not reach its backend "
    "and allowed the request — treat a sustained rate as an outage.",
    ["scope", "decision"],
)

ingest_duration_seconds = Histogram(
    "finvault_ingest_duration_seconds",
    "End-to-end document ingestion: load, chunk, embed, encrypt, store, extract.",
    buckets=_INGEST_BUCKETS,
)

documents_ingested_total = Counter(
    "finvault_documents_ingested_total",
    "Documents ingested by classification tier.",
    ["classification"],
)

compliance_verdicts_total = Counter(
    "finvault_compliance_verdicts_total",
    "Compliance outcomes. A spike in `blocked` with a stable question volume usually means "
    "a prompt or policy change, not an attack.",
    ["verdict", "reason"],
)


@contextmanager
def observe_duration(histogram: Any, **labels: str) -> Iterator[None]:
    """Times a block and records it even when the block raises.

    A `finally` rather than a normal return, because failures are usually
    slow ones — a timeout, a retry chain — and dropping them from the
    histogram makes latency look better precisely when it is worst.

    `time.monotonic()`, not `perf_counter()`: monotonic is the standard
    clock for durations at this scale (perf_counter's extra resolution is
    for micro-benchmarks, and these buckets start at a quarter-second), and
    using a different clock than the code being measured keeps
    instrumentation from perturbing it — the orchestrator's own execution
    trace times itself with `perf_counter`, and the two must not interleave.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        metric = histogram.labels(**labels) if labels else histogram
        metric.observe(time.monotonic() - start)


def record_tokens(*, agent: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Records token counts and, if rates are configured, the spend they imply.

    Cost is derived here rather than in the dashboard so that a rate change
    applies from the moment it is deployed, instead of silently re-pricing
    all of history the way a dashboard-side multiplier would.
    """
    from finvault.config import settings

    if input_tokens:
        llm_tokens_total.labels(agent=agent, model=model, direction="input").inc(input_tokens)
    if output_tokens:
        llm_tokens_total.labels(agent=agent, model=model, direction="output").inc(output_tokens)

    cost = (
        input_tokens * settings.finvault_price_input_per_million
        + output_tokens * settings.finvault_price_output_per_million
    ) / 1_000_000
    if cost:
        llm_cost_usd_total.labels(agent=agent, model=model).inc(cost)


def record_error(error_code: str, *, branch: str) -> None:
    errors_total.labels(error_code=error_code, branch=branch).inc()


def record_cache(cache_name: str, *, result: str) -> None:
    """`result` is one of hit / miss / store / unavailable."""
    cache_operations_total.labels(cache=cache_name, result=result).inc()


def render() -> tuple[bytes, str]:
    """The exposition payload and its content type, for GET /metrics."""
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "METRICS_AVAILABLE",
    "observe_duration",
    "record_tokens",
    "record_error",
    "record_cache",
    "render",
    "http_requests_total",
    "http_request_duration_seconds",
    "http_requests_in_flight",
    "errors_total",
    "agent_runs_total",
    "agent_duration_seconds",
    "llm_requests_total",
    "llm_tokens_total",
    "llm_cost_usd_total",
    "llm_retries_total",
    "token_budget_exhausted_total",
    "cache_operations_total",
    "rate_limit_decisions_total",
    "ingest_duration_seconds",
    "documents_ingested_total",
    "compliance_verdicts_total",
]
