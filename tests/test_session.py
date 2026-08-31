from __future__ import annotations

from finvault.agents.session import InMemorySessionStore


def test_append_and_get_history_roundtrip() -> None:
    store = InMemorySessionStore()
    store.append_turn(session_id="s1", user_id="u1", question="Q1", answer="A1")
    store.append_turn(session_id="s1", user_id="u1", question="Q2", answer="A2")

    history = store.get_history(session_id="s1", user_id="u1")

    assert [t.question for t in history] == ["Q1", "Q2"]
    assert [t.answer for t in history] == ["A1", "A2"]


def test_get_history_empty_for_unknown_session() -> None:
    store = InMemorySessionStore()
    assert store.get_history(session_id="nope", user_id="u1") == []


def test_history_isolated_between_users_even_for_the_same_session_id() -> None:
    """A session_id is not itself a secret or an ACL — isolation must come
    from also keying on the requesting user_id, so one user can never load
    another's history by guessing or reusing a session_id.
    """
    store = InMemorySessionStore()
    store.append_turn(session_id="shared", user_id="u1", question="Q1", answer="A1")

    assert store.get_history(session_id="shared", user_id="u2") == []
    assert len(store.get_history(session_id="shared", user_id="u1")) == 1
