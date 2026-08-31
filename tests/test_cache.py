"""Tests for finvault/cache.py.

The performance behavior of a cache is easy to eyeball and hard to get
wrong. What is easy to get wrong — and invisible when you do — is the key
scoping: a key missing the caller's clearance serves one tenant's answer to
another, and every test still passes because the cache "works". Those are
the assertions this file exists for.
"""

from __future__ import annotations

import time

import pytest

from finvault.cache import (
    InMemoryCache,
    bump_corpus_generation,
    corpus_generation,
    digest,
    scoped_key,
)


@pytest.fixture
def cache() -> InMemoryCache:
    return InMemoryCache()


# --- key construction: the security-relevant half ---


def test_the_same_question_in_two_orgs_produces_two_keys() -> None:
    a = scoped_key("answer", "what was Q3 revenue?", org_id="org-a", role="analyst")
    b = scoped_key("answer", "what was Q3 revenue?", org_id="org-b", role="analyst")
    assert a != b


def test_the_same_question_at_two_clearances_produces_two_keys() -> None:
    """The subtle one. Two users in the *same* org asking the identical
    question are entitled to different answers, because clearance decides
    what retrieval returned — so a key without the role is a clearance
    bypass for whoever asks second."""
    analyst = scoped_key("answer", "what was Q3 revenue?", org_id="org-a", role="analyst")
    officer = scoped_key("answer", "what was Q3 revenue?", org_id="org-a", role="compliance_officer")
    assert analyst != officer


def test_key_parts_cannot_collide_by_concatenation() -> None:
    """("ab","c") and ("a","bc") must not digest to the same key — one of
    those parts is an org id and the other is a question."""
    assert digest("ab", "c") != digest("a", "bc")


def test_a_key_does_not_reveal_the_content_it_indexes() -> None:
    """Keys are HMAC'd with a secret derived from the app's key material, so
    reading the cache does not let someone confirm a guessed document by
    hashing it themselves."""
    import hashlib

    question = "what was Q3 revenue?"
    key = scoped_key("answer", question, org_id="org-a", role="analyst")
    assert question not in key
    assert hashlib.sha256(question.encode()).hexdigest()[:40] not in key


def test_namespaces_separate_two_caches_of_the_same_value() -> None:
    assert scoped_key("answer", "x", org_id="o", role="r") != scoped_key("embedding", "x", org_id="o", role="r")


# --- storage semantics ---


def test_values_round_trip(cache: InMemoryCache) -> None:
    cache.set("k", {"answer": "42", "citations": []}, ttl_seconds=60)
    assert cache.get("k") == {"answer": "42", "citations": []}


def test_a_missing_key_is_none_not_an_error(cache: InMemoryCache) -> None:
    assert cache.get("never-written") is None


def test_an_expired_entry_reads_as_a_miss(cache: InMemoryCache) -> None:
    cache.set("k", "v", ttl_seconds=1)
    assert cache.get("k") == "v"
    # Reaching into the stored expiry rather than sleeping: a test that waits
    # a real second to prove expiry works is a test people stop running.
    value, _ = cache._store["k"]
    cache._store["k"] = (value, time.time() - 1)
    assert cache.get("k") is None


def test_delete_removes_an_entry(cache: InMemoryCache) -> None:
    cache.set("k", "v", ttl_seconds=60)
    cache.delete("k")
    assert cache.get("k") is None


# --- counters, which rate limiting depends on ---


def test_incr_counts_up_from_one(cache: InMemoryCache) -> None:
    assert [cache.incr("c", ttl_seconds=60) for _ in range(3)] == [1, 2, 3]


def test_incr_does_not_extend_the_window_it_is_counting(cache: InMemoryCache) -> None:
    """A fixed window whose expiry slid forward on every request would never
    close under sustained load — precisely when the limit has to hold."""
    cache.incr("c", ttl_seconds=60)
    _, first_expiry = cache._store["c"]
    cache.incr("c", ttl_seconds=60)
    _, second_expiry = cache._store["c"]
    assert first_expiry == second_expiry


def test_a_counter_restarts_after_its_window_closes(cache: InMemoryCache) -> None:
    cache.incr("c", ttl_seconds=60)
    count, _ = cache._store["c"]
    cache._store["c"] = (count, time.time() - 1)
    assert cache.incr("c", ttl_seconds=60) == 1


# --- corpus generation: the invalidation mechanism ---


def test_generation_starts_at_zero_and_increments(cache: InMemoryCache) -> None:
    assert corpus_generation(cache, "org-a") == 0
    assert bump_corpus_generation(cache, "org-a") == 1
    assert corpus_generation(cache, "org-a") == 1


def test_a_bump_retires_every_cached_answer_for_that_org(cache: InMemoryCache) -> None:
    """TTL alone cannot do this: a user who uploads a document and
    immediately asks about it must not be served an answer computed before
    the upload, however short the TTL."""

    def key() -> str:
        return scoped_key("answer", "q", str(corpus_generation(cache, "org-a")), org_id="org-a", role="analyst")

    before = key()
    cache.set(before, "stale answer", ttl_seconds=3600)
    assert cache.get(before) == "stale answer"

    bump_corpus_generation(cache, "org-a")

    after = key()
    assert after != before
    # The stale entry is not deleted — it is simply unreachable, and expires
    # on its own. No scan, no delete, no key enumeration.
    assert cache.get(after) is None


def test_one_orgs_bump_does_not_invalidate_anothers(cache: InMemoryCache) -> None:
    bump_corpus_generation(cache, "org-a")
    assert corpus_generation(cache, "org-a") == 1
    assert corpus_generation(cache, "org-b") == 0


# --- Redis, when one is reachable ---


@pytest.mark.redis
def test_redis_backend_round_trips_and_counts() -> None:
    """Skips itself without a server, so a developer with no Docker running
    still gets a green suite; CI runs it with a service container."""
    from finvault.cache import RedisCache

    try:
        cache = RedisCache("redis://localhost:6379/15", socket_timeout=0.5)
    except Exception as exc:
        pytest.skip(f"no Redis available: {exc}")
    if not cache.available:
        pytest.skip("no Redis available")

    cache.delete("fv:test:roundtrip")
    cache.delete("fv:test:counter")

    cache.set("fv:test:roundtrip", {"a": 1}, ttl_seconds=30)
    assert cache.get("fv:test:roundtrip") == {"a": 1}
    assert [cache.incr("fv:test:counter", ttl_seconds=30) for _ in range(3)] == [1, 2, 3]

    cache.delete("fv:test:roundtrip")
    cache.delete("fv:test:counter")
