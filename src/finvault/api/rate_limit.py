"""Per-identity request rate limiting.

`POST /query` is the endpoint that needs this. One call fans out into
several nested LLM calls — Orchestrator, the Retriever's own multi-round
search loop, Analyst, and the Compliance semantic review — so what a client
experiences as one request is a multiple of that in provider spend and
upstream quota. Without a limit, a single looping client exhausts the day's
model budget for every tenant, which is an availability failure dressed up
as a cost problem.

**Two windows, both enforced.** A per-minute window catches bursts; a
per-hour window catches the slow drip that stays under the burst limit all
day. Passing one and failing the other is a rejection — a limiter with only
a short window is trivially defeated by pacing.

**Identity, not connection.** Authenticated callers are limited by user id,
so a limit follows the account across browser tabs, devices, and IPs rather
than being reset by any of them. Only unauthenticated requests fall back to
the client IP, and get a much tighter allowance, because there is no
account behind them to hold accountable. IP is used *only* in that fallback:
NAT and corporate egress routinely put hundreds of legitimate users behind
one address, so limiting authenticated traffic by IP would penalize exactly
the enterprise customers this system is built for.

**This layer fails open, and that is deliberate.** Everywhere else in
FinVault a broken dependency means refuse the request. Here the dependency
is a counter in Redis, and refusing every request because the counter is
unreachable converts a cache outage into a total outage — the limiter would
become the very denial of service it exists to prevent. A `failed_open`
metric makes the condition visible, and the window is short enough that the
exposure is bounded. If a deployment needs the opposite, it is one branch in
`_check_window()`, marked below.
"""

from __future__ import annotations

from dataclasses import dataclass

from starlette.datastructures import Headers

from finvault.cache import Cache, digest
from finvault.config import settings
from finvault.errors import RateLimitExceeded
from finvault.metrics import rate_limit_decisions_total
from finvault.observability import extra_fields, get_logger

logger = get_logger(__name__)

# Paths a limiter must never sit in front of. /health is polled by the
# orchestrator's liveness probe — rate-limiting it would let a burst of user
# traffic make Kubernetes restart a healthy pod. /metrics is scraped on a
# fixed interval by one client, so a limit there only ever breaks monitoring
# at the moment monitoring matters.
EXEMPT_PATHS = frozenset({"/health", "/metrics"})


@dataclass(frozen=True)
class Window:
    """One limit: `limit` requests per `seconds`."""

    name: str
    limit: int
    seconds: int


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    window: Window | None = None
    count: int = 0
    retry_after_seconds: int = 0


class RateLimiter:
    """Fixed-window counters in the shared cache.

    Fixed windows, not a sliding log: a sliding window needs one stored
    entry per request, which is a large multiple of the memory and the
    network round-trips for a bound that differs only at the window edge.
    The known imprecision is that a caller can send `limit` requests at the
    end of one window and `limit` again at the start of the next. With the
    burst window at a minute and the sustained window at an hour, that edge
    case cannot produce sustained overuse — the hourly window closes over it.
    """

    def __init__(self, cache: Cache, *, windows: list[Window] | None = None) -> None:
        self._cache = cache
        self._windows = windows or [
            Window("minute", settings.finvault_rate_limit_per_minute, 60),
            Window("hour", settings.finvault_rate_limit_per_hour, 3600),
        ]

    def check(self, identity: str, *, scope: str) -> Verdict:
        """Counts this request against every window and returns the first
        verdict that denies it.

        Every window is incremented even when an earlier one already denied,
        so a client held off by the minute window still accrues against the
        hour — otherwise a caller could sit permanently at the burst limit
        and never accumulate the sustained count that is supposed to catch
        exactly that.
        """
        denial: Verdict | None = None
        for window in self._windows:
            verdict = self._check_window(identity, window, scope=scope)
            if not verdict.allowed and denial is None:
                denial = verdict

        if denial is not None:
            rate_limit_decisions_total.labels(scope=scope, decision="denied").inc()
            logger.warning(
                "rate_limit_exceeded",
                extra=extra_fields(
                    scope=scope,
                    window=denial.window.name if denial.window else None,
                    limit=denial.window.limit if denial.window else None,
                    count=denial.count,
                    retry_after_seconds=denial.retry_after_seconds,
                ),
            )
            return denial

        rate_limit_decisions_total.labels(scope=scope, decision="allowed").inc()
        return Verdict(allowed=True)

    def _check_window(self, identity: str, window: Window, *, scope: str) -> Verdict:
        key = f"fv:rl:{window.name}:{digest(scope, identity)}"
        count = self._cache.incr(key, ttl_seconds=window.seconds)

        if count < 0:
            # The counter is unknown — the cache backend is unreachable.
            # FAIL OPEN. To fail closed instead, return a denying Verdict
            # here; read this module's docstring first, because doing so
            # turns a cache outage into a full outage.
            rate_limit_decisions_total.labels(scope=scope, decision="failed_open").inc()
            return Verdict(allowed=True)

        if count > window.limit:
            return Verdict(allowed=False, window=window, count=count, retry_after_seconds=window.seconds)
        return Verdict(allowed=True, window=window, count=count)


def identify(request_headers: Headers, *, client_host: str | None, actor: str | None) -> tuple[str, str]:
    """Returns `(identity, scope)` for the caller.

    `scope` separates the two counter namespaces so an authenticated user
    and an anonymous client from the same address never share a counter.

    The IP is taken from the socket, not from `X-Forwarded-For`. That header
    is client-supplied and trivially spoofed: honoring it would let one
    caller mint unlimited identities by varying it, which is worse than no
    limit at all. Behind a proxy, configure the proxy to *set* (not append)
    a trusted header and read it here — that is a deployment-specific change
    and is deliberately not guessed at.
    """
    if actor:
        return actor, "user"
    return client_host or "unknown", "anonymous"


def enforce(limiter: RateLimiter, *, identity: str, scope: str) -> None:
    """Raises `RateLimitExceeded` when the caller is over a limit.

    `retry_after_seconds` travels in the exception's `context`, which
    `api/error_handlers.py` turns into the `Retry-After` header — so a
    client is told how long to wait rather than having to guess.
    """
    verdict = limiter.check(identity, scope=scope)
    if verdict.allowed:
        return

    window = verdict.window
    raise RateLimitExceeded(
        f"Rate limit exceeded: {verdict.count} requests in the {window.name if window else '?'} window",
        context={
            "scope": scope,
            "window": window.name if window else None,
            "limit": window.limit if window else None,
            "retry_after_seconds": verdict.retry_after_seconds,
        },
        user_message=(
            f"Too many requests — the limit is {window.limit} per {window.name}. "
            f"Try again in {verdict.retry_after_seconds} seconds."
            if window
            else RateLimitExceeded.user_message
        ),
    )


def anonymous_limiter(cache: Cache) -> RateLimiter:
    """A tighter limiter for requests with no verified identity."""
    return RateLimiter(
        cache,
        windows=[
            Window("minute", settings.finvault_rate_limit_anonymous_per_minute, 60),
            Window("hour", settings.finvault_rate_limit_anonymous_per_minute * 20, 3600),
        ],
    )


__all__ = ["RateLimiter", "Window", "Verdict", "identify", "enforce", "anonymous_limiter", "EXEMPT_PATHS"]
