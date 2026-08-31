"""Tests for api/rate_limit.py.

Two things need pinning here. The first is arithmetic: the limit holds, both
windows are enforced, and the counter is per identity. The second is the
policy decision this module makes that nothing else in FinVault makes — it
fails *open* when its backend is unreachable, and a test has to say so out
loud, because "fail closed" is the rule everywhere else and a future reader
will otherwise assume this file forgot.
"""

from __future__ import annotations

import pytest
from starlette.datastructures import Headers

from finvault.api.rate_limit import EXEMPT_PATHS, RateLimiter, Window, enforce, identify
from finvault.cache import Cache, InMemoryCache
from finvault.errors import RateLimitExceeded


class UnreachableCache(InMemoryCache):
    """A cache whose counters are unavailable — what a Redis outage looks
    like from the limiter's side (see RedisCache.incr, which returns -1)."""

    def incr(self, key: str, *, ttl_seconds: int) -> int:
        return -1


def _limiter(cache: Cache | None = None, *, limit: int = 3, seconds: int = 60) -> RateLimiter:
    return RateLimiter(cache or InMemoryCache(), windows=[Window("minute", limit, seconds)])


def test_requests_up_to_the_limit_are_allowed() -> None:
    limiter = _limiter(limit=3)
    assert [limiter.check("u1", scope="user").allowed for _ in range(3)] == [True, True, True]


def test_the_request_past_the_limit_is_denied() -> None:
    limiter = _limiter(limit=3)
    for _ in range(3):
        limiter.check("u1", scope="user")
    verdict = limiter.check("u1", scope="user")
    assert verdict.allowed is False
    assert verdict.retry_after_seconds == 60


def test_counters_are_per_identity() -> None:
    """A limit that one busy user could impose on everyone else would be a
    denial of service with extra steps."""
    limiter = _limiter(limit=2)
    for _ in range(2):
        limiter.check("u1", scope="user")
    assert limiter.check("u1", scope="user").allowed is False
    assert limiter.check("u2", scope="user").allowed is True


def test_an_authenticated_user_and_an_anonymous_client_never_share_a_counter() -> None:
    limiter = _limiter(limit=1)
    limiter.check("same-string", scope="user")
    assert limiter.check("same-string", scope="user").allowed is False
    assert limiter.check("same-string", scope="anonymous").allowed is True


def test_both_windows_are_enforced_and_the_denial_names_the_one_that_tripped() -> None:
    """A limiter with only a burst window is defeated by pacing; one with
    only a sustained window lets a burst straight through."""
    limiter = RateLimiter(InMemoryCache(), windows=[Window("minute", 10, 60), Window("hour", 3, 3600)])
    for _ in range(3):
        assert limiter.check("u1", scope="user").allowed is True

    verdict = limiter.check("u1", scope="user")
    assert verdict.allowed is False
    # Under the minute limit, over the hour limit.
    assert verdict.window is not None and verdict.window.name == "hour"


def test_a_denied_request_still_accrues_against_the_other_window() -> None:
    """Otherwise a client could sit permanently at the burst limit and never
    accumulate the sustained count that exists to catch exactly that."""
    cache = InMemoryCache()
    limiter = RateLimiter(cache, windows=[Window("minute", 1, 60), Window("hour", 100, 3600)])
    for _ in range(5):
        limiter.check("u1", scope="user")

    hour_counters = [v for k, (v, _) in cache._store.items() if ":hour:" in k]
    assert hour_counters == [5]


def test_the_limiter_fails_open_when_its_backend_is_unreachable() -> None:
    """Deliberate, and the one place FinVault does not fail closed: refusing
    every request because a counter is unreachable turns a cache outage into
    a total outage — the limiter would become the denial of service it
    exists to prevent. See the module docstring."""
    limiter = _limiter(UnreachableCache(), limit=1)
    assert [limiter.check("u1", scope="user").allowed for _ in range(10)] == [True] * 10


def test_failing_open_is_counted_so_the_condition_is_visible() -> None:
    """Failing open silently would be indistinguishable from having no limit
    configured at all."""
    from prometheus_client import REGISTRY

    def failed_open_count() -> float:
        return (
            REGISTRY.get_sample_value(
                "finvault_rate_limit_decisions_total", {"scope": "metrics-probe", "decision": "failed_open"}
            )
            or 0.0
        )

    before = failed_open_count()
    _limiter(UnreachableCache()).check("u1", scope="metrics-probe")
    assert failed_open_count() == before + 1


# --- enforce(): the exception the API layer turns into a 429 ---


def test_enforce_raises_with_a_retry_after_the_handler_can_use() -> None:
    limiter = _limiter(limit=1)
    enforce(limiter, identity="u1", scope="user")

    with pytest.raises(RateLimitExceeded) as exc_info:
        enforce(limiter, identity="u1", scope="user")

    exc = exc_info.value
    assert exc.http_status == 429
    # Read by api/error_handlers.py to set the Retry-After header — a 429
    # without one leaves the client guessing, and the usual guess is
    # "immediately".
    assert exc.context["retry_after_seconds"] == 60
    assert exc.retryable is True


def test_the_message_tells_the_caller_the_limit_and_the_wait() -> None:
    limiter = _limiter(limit=1)
    enforce(limiter, identity="u1", scope="user")
    with pytest.raises(RateLimitExceeded) as exc_info:
        enforce(limiter, identity="u1", scope="user")
    message = exc_info.value.user_message
    assert "1 per minute" in message
    assert "60 seconds" in message


# --- identity selection ---


def test_an_authenticated_caller_is_limited_by_account_not_address() -> None:
    """So the limit follows the account across tabs, devices and IPs rather
    than being reset by any of them."""
    identity, scope = identify(Headers({}), client_host="10.0.0.1", actor="user-42")
    assert (identity, scope) == ("user-42", "user")


def test_only_unauthenticated_callers_fall_back_to_the_client_address() -> None:
    """NAT and corporate egress put many legitimate users behind one address,
    so IP is the fallback of last resort, never the primary identity."""
    identity, scope = identify(Headers({}), client_host="10.0.0.1", actor=None)
    assert (identity, scope) == ("10.0.0.1", "anonymous")


def test_a_caller_with_no_address_at_all_still_gets_a_bucket() -> None:
    identity, scope = identify(Headers({}), client_host=None, actor=None)
    assert (identity, scope) == ("unknown", "anonymous")


def test_health_and_metrics_are_exempt() -> None:
    """Rate-limiting /health would let a burst of user traffic make
    Kubernetes restart a healthy pod; rate-limiting /metrics breaks
    monitoring at the moment monitoring matters."""
    assert "/health" in EXEMPT_PATHS
    assert "/metrics" in EXEMPT_PATHS
