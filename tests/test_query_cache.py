"""Tests for the query-answer cache on POST /query (api/routes.py).

Caching answers in an ACL-gated multi-tenant system is the highest-risk
cache in this codebase: every bug here is a disclosure, not a slowdown. The
tests are written from that angle — what must *not* be served from cache
gets more coverage than what must.

The route function is called directly, matching the rest of this codebase's
route-testing approach (see test_compare_route.py's docstring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from finvault.agents.orchestrator import OrchestratorResult
from finvault.api.routes import QueryRequest, _cacheable, _query_cache_key, query
from finvault.cache import InMemoryCache, bump_corpus_generation
from finvault.config import settings
from finvault.models import Role, User
from finvault.security.audit import InMemoryAuditLog


class RecordingOrchestrator:
    """Counts how many times the pipeline actually ran, which is the only
    direct evidence a cache hit occurred."""

    def __init__(self, answer: str = "Q3 revenue was $10 million.") -> None:
        self.calls = 0
        self._answer = answer

    def handle(self, question: str, *, session_id=None, event_bus=None) -> OrchestratorResult:
        self.calls += 1
        return OrchestratorResult(
            answer=self._answer,
            blocked=False,
            block_reason=None,
            citations=[{"document": "10-K", "quoted_text": "revenue was $10 million"}],
        )


@pytest.fixture
def env(monkeypatch):
    cache = InMemoryCache()
    audit_log = InMemoryAuditLog()
    orchestrator = RecordingOrchestrator()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(cache=cache, audit_log=audit_log)))
    monkeypatch.setattr("finvault.api.routes._build_orchestrator", lambda *_args, **_kwargs: orchestrator)
    monkeypatch.setattr(settings, "finvault_enable_query_cache", True)
    return SimpleNamespace(cache=cache, audit_log=audit_log, orchestrator=orchestrator, request=request)


def _user(role: Role = Role.ANALYST, org: str = "org-a") -> User:
    return User(username=f"u-{role.value}", role=role, org_id=org)


def test_a_repeated_question_is_answered_from_cache(env) -> None:
    user = _user()
    first = query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    second = query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)

    assert first.answer == second.answer
    assert env.orchestrator.calls == 1


def test_two_orgs_never_share_an_answer(env) -> None:
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=_user(org="org-a"))
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=_user(org="org-b"))
    assert env.orchestrator.calls == 2


def test_two_clearances_in_one_org_never_share_an_answer(env) -> None:
    """The failure this test exists for: retrieval is clearance-filtered, so
    an answer computed for a compliance officer can contain content an
    analyst is not cleared to see. Sharing the cache entry would be a
    clearance bypass that leaves no trace."""
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=_user(Role.ANALYST))
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=_user(Role.COMPLIANCE_OFFICER))
    assert env.orchestrator.calls == 2


def test_ingesting_a_document_retires_the_orgs_cached_answers(env) -> None:
    """A user who uploads a policy and immediately asks about it must not be
    told what was true before the upload."""
    user = _user()
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert env.orchestrator.calls == 1

    bump_corpus_generation(env.cache, user.org_id)

    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert env.orchestrator.calls == 2


def test_another_orgs_ingest_does_not_retire_this_orgs_answers(env) -> None:
    user = _user(org="org-a")
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    bump_corpus_generation(env.cache, "org-b")
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert env.orchestrator.calls == 1


def test_a_conversation_turn_is_never_cached(env) -> None:
    """The same question in two conversations has two correct answers, and
    the key does not encode history. Caching a follow-up like "and the year
    before?" would answer it from someone else's conversation."""
    user = _user()
    query(QueryRequest(question="And the year before?", session_id="s-1"), env.request, user=user)
    query(QueryRequest(question="And the year before?", session_id="s-2"), env.request, user=user)
    assert env.orchestrator.calls == 2


def test_a_blocked_answer_is_never_cached() -> None:
    """A compliance block is a decision about one request. It is cheap to
    recompute and must be re-decided, not replayed."""
    blocked = OrchestratorResult(answer="Blocked.", blocked=True, block_reason="policy", citations=[])
    assert _cacheable(QueryRequest(question="q"), blocked) is False


def test_an_answer_with_no_citations_is_never_cached() -> None:
    """No citations means the model answered without calling analyze — a
    degraded path (see orchestrator.py). Caching it makes one bad turn
    durable."""
    degraded = OrchestratorResult(answer="Probably $10M.", blocked=False, block_reason=None, citations=[])
    assert _cacheable(QueryRequest(question="q"), degraded) is False


def test_a_clean_cited_answer_to_a_sessionless_question_is_cacheable() -> None:
    good = OrchestratorResult(answer="$10M.", blocked=False, block_reason=None, citations=[{"quoted_text": "$10M"}])
    assert _cacheable(QueryRequest(question="q"), good) is True


def test_a_cached_answer_is_still_audit_logged(env) -> None:
    """A cached answer is still a disclosure of document-derived content to
    this user at this moment. An audit log that only recorded uncached
    queries would under-record exactly the repeated access patterns an
    auditor looks for."""
    user = _user()
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)

    actions = [entry.action for entry in env.audit_log.entries()]
    assert "query_cached" in actions


def test_the_execution_trace_is_not_replayed_from_cache(env) -> None:
    """The trace records what *this* request did, with real measured
    durations. Replaying it would present fabricated timings as
    measurements — the exact thing orchestrator.py's _run_step exists to
    avoid."""
    user = _user()
    query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    cached = query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert cached.execution_steps == []


def test_each_response_carries_its_own_session_id_even_on_a_hit(env) -> None:
    """The session id identifies this exchange, not the cached content — two
    users hitting the same entry must not be handed one another's."""
    user = _user()
    first = query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    second = query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert first.session_id != second.session_id


def test_disabling_the_cache_makes_every_query_run_the_pipeline(env, monkeypatch) -> None:
    monkeypatch.setattr(settings, "finvault_enable_query_cache", False)
    user = _user()
    for _ in range(3):
        query(QueryRequest(question="What was Q3 revenue?"), env.request, user=user)
    assert env.orchestrator.calls == 3


def test_whitespace_is_normalized_but_case_and_punctuation_are_not(env) -> None:
    """Trimming is safe. Folding case or punctuation is not: both change what
    an embedding retrieves, so treating them as the same question would serve
    an answer to a different one."""
    cache = env.cache
    user = _user()
    base = _query_cache_key("What was Q3 revenue?", user=user, cache=cache)
    assert _query_cache_key("  What was Q3 revenue?  ", user=user, cache=cache) == base
    assert _query_cache_key("what was q3 revenue?", user=user, cache=cache) != base
    assert _query_cache_key("What was Q3 revenue", user=user, cache=cache) != base
