"""Per-session conversation memory.

Only compliance-approved answers ever enter history — a blocked turn's
"answer" is a generic refusal message anyway, and every stored answer has
already passed classification/PII/citation checks once (see
orchestrator.py), so replaying it into a later turn's context doesn't
introduce new leakage beyond what already happened once.

Keyed by (user_id, session_id) so one user can never load another's history
by guessing or reusing a session_id — the same non-inference posture as
retrieval's per-user ACL filtering (retrieval/retriever.py).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from finvault.db import sessions_table


@dataclass(frozen=True)
class SessionTurn:
    question: str
    answer: str
    timestamp: float


class SessionStore(ABC):
    @abstractmethod
    def append_turn(self, *, session_id: str, user_id: str, question: str, answer: str) -> None: ...

    @abstractmethod
    def get_history(self, *, session_id: str, user_id: str) -> list[SessionTurn]: ...


class InMemorySessionStore(SessionStore):
    """Reference implementation — not persisted across process restarts. A
    production deployment would back this with Postgres, the same
    InMemory/Postgres split already used for AuditLog (security/audit.py).
    """

    def __init__(self) -> None:
        self._turns: dict[str, list[SessionTurn]] = {}

    @staticmethod
    def _key(*, session_id: str, user_id: str) -> str:
        return f"{user_id}:{session_id}"

    def append_turn(self, *, session_id: str, user_id: str, question: str, answer: str) -> None:
        key = self._key(session_id=session_id, user_id=user_id)
        self._turns.setdefault(key, []).append(SessionTurn(question=question, answer=answer, timestamp=time.time()))

    def get_history(self, *, session_id: str, user_id: str) -> list[SessionTurn]:
        return list(self._turns.get(self._key(session_id=session_id, user_id=user_id), []))


class PostgresSessionStore(SessionStore):
    """Persistent session store backed by the `sessions` table (see db.py).
    Survives process restarts and is shared correctly across multiple
    workers — the two problems InMemorySessionStore has in a real
    deployment.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append_turn(self, *, session_id: str, user_id: str, question: str, answer: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    created_at=time.time(),
                )
            )

    def get_history(self, *, session_id: str, user_id: str) -> list[SessionTurn]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(sessions_table)
                .where(sessions_table.c.session_id == session_id, sessions_table.c.user_id == user_id)
                .order_by(sessions_table.c.id.asc())
            ).all()
        return [SessionTurn(question=row.question, answer=row.answer, timestamp=row.created_at) for row in rows]
