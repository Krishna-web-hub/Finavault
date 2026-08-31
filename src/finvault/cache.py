"""Cache interface, with a Redis implementation and an in-memory reference.

Same shape as every other infra dependency here (see
retrieval/vector_store.py, security/encryption.py): an interface the code
depends on, a real backend for deployment, and an in-memory equivalent so
the whole system runs and tests without the infra.

**Cache keys in a multi-tenant system are a security surface, not a
performance detail.** Two rules hold everywhere in this file, and any new
cached value must obey both:

1. *A key must encode everything that changes the answer.* For anything
   derived from documents that means the org **and** the caller's role — an
   analyst and a compliance officer asking the identical question are
   entitled to different answers, because clearance filters what was
   retrieved. A key missing the role would serve a restricted-tier answer
   to whoever asked second. `scoped_key()` exists so no call site builds
   one by hand.
2. *A key must never be a plaintext-recoverable digest of content.* Keys
   are HMAC'd with a secret derived from the app's own key material
   (`_key_secret()`), so someone who reads Redis cannot confirm a guessed
   document's presence by hashing it themselves — the unkeyed-digest
   trade-off `db.py` makes deliberately for `label_hash` is not repeated
   here, because a cache is far more likely to be exposed than the primary
   database.

Values are stored as JSON, never pickle: a cache is a remote, mutable store,
and `pickle.loads` on anything an attacker can write is remote code
execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from finvault.config import settings
from finvault.observability import extra_fields, get_logger, log_exception

logger = get_logger(__name__)

# Namespace prefix on every key, so FinVault's entries are identifiable (and
# separately flushable) in a Redis instance that isn't exclusively ours.
KEY_PREFIX = "fv"


def _key_secret() -> bytes:
    """Secret for key HMACs. Derived from the JWT secret rather than adding
    another one to configure: it is already required, already deployment
    -specific, and already must be rotated to invalidate outstanding trust.
    Rotating it also invalidates every cache key, which is the correct
    behavior — cached content derived under old key material should not
    survive the rotation.
    """
    return hashlib.sha256(f"finvault-cache::{settings.finvault_jwt_secret}".encode()).digest()


def digest(*parts: str) -> str:
    """Keyed digest of the parts, joined unambiguously.

    The `\\x1f` separator (ASCII unit separator) prevents the collision a
    plain concatenation invites: ("ab", "c") and ("a", "bc") must not
    produce the same key when one part is an org id and the other is a
    question.
    """
    message = "\x1f".join(parts).encode()
    return hmac.new(_key_secret(), message, hashlib.sha256).hexdigest()[:40]


def scoped_key(namespace: str, *parts: str, org_id: str, role: str | None = None) -> str:
    """The only supported way to build a cache key for tenant-derived data.

    Putting `org_id` and `role` in the digest — rather than trusting call
    sites to remember — is what makes a cross-tenant or cross-clearance hit
    structurally impossible rather than merely unlikely.
    """
    return f"{KEY_PREFIX}:{namespace}:{digest(org_id, role or '-', *parts)}"


class Cache(ABC):
    """Get/set/delete plus the two atomic primitives rate limiting needs.

    Every method must **fail soft**: a cache that is down degrades the
    system to "no cache", never to an error. This is the one place in the
    codebase that deliberately does not fail closed — a cache miss and a
    cache outage produce the same correct (if slower) result, so there is
    nothing to fail closed about. Rate limiting is the exception and states
    its own policy at its call site.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def incr(self, key: str, *, ttl_seconds: int) -> int:
        """Atomically increments a counter, setting its TTL on first use.
        Returns the value after incrementing. Atomicity is required: rate
        limiting with a read-modify-write would let concurrent requests each
        read the same count and all pass.
        """

    @abstractmethod
    def bump(self, key: str) -> int:
        """Atomically increments a generation counter that never expires.
        Used to invalidate whole families of keys at once — see
        `corpus_generation()` below.
        """

    @property
    @abstractmethod
    def available(self) -> bool:
        """False when the backend is unreachable. Callers use it for
        reporting (health, metrics), never to decide whether to try — the
        methods above already degrade on their own.
        """


class InMemoryCache(Cache):
    """Process-local reference implementation. Correct, and genuinely useful
    for a single-process deployment or a test — but not shared between
    workers, so a multi-replica deployment gets one cache per replica and a
    correspondingly lower hit rate. It is not a substitute for Redis in
    production; it is what keeps the system runnable without it.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _live(self, key: str) -> tuple[Any, float | None] | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        _, expires_at = entry
        if expires_at is not None and expires_at <= time.time():
            # Lazy expiry: entries are evicted when looked at, not by a
            # sweeper. Adequate here because this backend exists for tests
            # and single-process runs, where the working set is small.
            self._store.pop(key, None)
            return None
        return entry

    def get(self, key: str) -> Any | None:
        entry = self._live(key)
        return None if entry is None else entry[0]

    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        self._store[key] = (value, time.time() + ttl_seconds if ttl_seconds > 0 else None)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def incr(self, key: str, *, ttl_seconds: int) -> int:
        entry = self._live(key)
        if entry is None:
            self._store[key] = (1, time.time() + ttl_seconds)
            return 1
        count, expires_at = entry
        # The existing expiry is kept, not extended — a fixed window that
        # slid forward on every request would never close under sustained
        # load, which is precisely when the limit has to hold.
        self._store[key] = (count + 1, expires_at)
        return count + 1

    def bump(self, key: str) -> int:
        # Stored in the same dict, with no expiry, so `corpus_generation()`
        # reads it back through the ordinary `get()` path — a generation
        # kept in a separate structure would be invisible to that reader,
        # and the invalidation would silently never happen.
        current = self.get(key)
        value = (current if isinstance(current, int) else 0) + 1
        self._store[key] = (value, None)
        return value

    @property
    def available(self) -> bool:
        return True


class RedisCache(Cache):
    """Redis-backed shared cache.

    Constructed lazily and never raises on connection failure: a deployment
    whose Redis is down must keep serving. Every operation catches
    `RedisError` and degrades to a miss, logged at WARNING so the outage is
    visible without one record per request.
    """

    def __init__(self, url: str, *, socket_timeout: float = 0.25) -> None:
        import redis  # deferred: optional dependency, only needed when configured

        self._url = url
        self._error = redis.RedisError
        # Tight timeouts on purpose. A cache is an optimization sitting in
        # the request path; if it cannot answer in a fraction of the time the
        # real work takes, waiting for it is strictly worse than missing.
        self._client = redis.Redis.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
            health_check_interval=30,
        )
        self._available = False
        self.ping()

    def ping(self) -> bool:
        try:
            self._client.ping()
            self._available = True
        except self._error as exc:
            self._available = False
            logger.warning(
                "cache_unavailable",
                extra=extra_fields(backend="redis", url=self._url, error_message=str(exc)),
            )
        return self._available

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except self._error as exc:
            self._degraded("get", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Someone else's key collided with our namespace, or an entry
            # was written by an older schema. Drop it rather than raising:
            # an unreadable cache entry is a miss, not a request failure.
            logger.warning("cache_entry_unreadable", extra=extra_fields(key=key))
            self.delete(key)
            return None

    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        try:
            self._client.set(key, json.dumps(value, default=str), ex=ttl_seconds if ttl_seconds > 0 else None)
            self._available = True
        except self._error as exc:
            self._degraded("set", exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except self._error as exc:
            self._degraded("delete", exc)

    def incr(self, key: str, *, ttl_seconds: int) -> int:
        try:
            # A pipeline, so INCR and EXPIRE reach the server together. NX on
            # the expire means the window is set once, when the counter is
            # created, and is not pushed forward by later requests inside it.
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_seconds, nx=True)
            count, _ = pipe.execute()
            self._available = True
            return int(count)
        except self._error as exc:
            self._degraded("incr", exc)
            # -1 tells the caller the counter is unknown. The rate limiter
            # decides what that means; see api/rate_limit.py, which fails
            # OPEN here and says why.
            return -1

    def bump(self, key: str) -> int:
        try:
            return int(self._client.incr(key))
        except self._error as exc:
            self._degraded("bump", exc)
            return -1

    @property
    def available(self) -> bool:
        return self._available

    def _degraded(self, operation: str, exc: Exception) -> None:
        was_available = self._available
        self._available = False
        if was_available:
            # Logged on the transition only. A hard-down Redis in the request
            # path would otherwise emit one record per operation per request.
            log_exception(logger, exc, "cache_operation_failed", operation=operation, backend="redis")


def build_cache() -> Cache:
    """The cache this deployment should use, chosen from configuration.

    Returns `InMemoryCache` when no Redis URL is set, or when Redis is
    configured but unreachable at startup — so a missing or broken cache
    degrades performance and never availability.
    """
    if not settings.finvault_redis_url:
        logger.info("cache_backend_selected", extra=extra_fields(backend="in_memory", reason="no_redis_url"))
        return InMemoryCache()

    try:
        cache = RedisCache(settings.finvault_redis_url, socket_timeout=settings.finvault_redis_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 — a bad URL or missing driver must not stop startup
        log_exception(logger, exc, "cache_backend_failed_falling_back", backend="redis")
        return InMemoryCache()

    if not cache.available:
        logger.warning(
            "cache_backend_selected",
            extra=extra_fields(backend="in_memory", reason="redis_unreachable", url=settings.finvault_redis_url),
        )
        return InMemoryCache()

    logger.info("cache_backend_selected", extra=extra_fields(backend="redis", url=settings.finvault_redis_url))
    return cache


# --- Corpus generation: invalidating an org's cached answers on ingest ---


def _generation_key(org_id: str) -> str:
    return f"{KEY_PREFIX}:gen:{digest(org_id)}"


def corpus_generation(cache: Cache, org_id: str) -> int:
    """The org's current corpus version, included in every cached answer's
    key.

    Time-based expiry alone is not enough for a document store: a user who
    uploads a policy and immediately asks about it must not be served a
    cached answer that predates the upload, however short the TTL. Bumping a
    counter on ingest changes every subsequent key for that org, which
    retires the whole family at once without scanning or deleting anything —
    the stale entries simply become unreachable and expire on their own.
    """
    value = cache.get(_generation_key(org_id))
    return int(value) if isinstance(value, int) else 0


def bump_corpus_generation(cache: Cache, org_id: str) -> int:
    """Called after anything that changes what a query could retrieve."""
    generation = cache.bump(_generation_key(org_id))
    logger.info("corpus_generation_bumped", extra=extra_fields(org_id=org_id, generation=generation))
    return generation


__all__ = [
    "Cache",
    "InMemoryCache",
    "RedisCache",
    "build_cache",
    "scoped_key",
    "digest",
    "corpus_generation",
    "bump_corpus_generation",
    "KEY_PREFIX",
]
