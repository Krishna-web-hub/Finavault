"""Per-document quarantine — the response half of injection detection.

`detect_injection_attempt` (security/guardrails.py) has always been able to
spot a poisoned chunk, and ComplianceAgent now blocks on it. But detection
without a response left an operator with nowhere to go: the offending
document stayed in the vector store and was retrieved again on the next
semantically-similar query, blocking that one too. This store is the
response — a compliance officer marks the document quarantined and the
Retriever stops returning it.

Quarantine is a *deliberate human action*, never automatic. A heuristic
false-positive (a legitimate policy document that quotes "ignore previous
instructions" as an example of what to watch for) would otherwise silently
remove a real document from the corpus, and a silent retrieval gap is
exactly the failure mode this codebase avoids elsewhere — see
retriever_agent.py, which withholds over-classified content with an
explicit marker rather than dropping it.

Scoped per document, not per chunk: an attacker who can place text in a
document can split a payload across chunk boundaries, so chunk-level
quarantine is trivially evaded by the party it exists to stop.

Release is a status change, not a delete, so the quarantine decision itself
stays on the record — the same append-only posture as review_queue.py.
Note that quarantine hides a document from retrieval; it does not erase it.
Erasure is a separate, unsolved problem (no delete path exists in this
codebase for Qdrant, the graph store, or the audit log).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from finvault.db import quarantined_documents_table

QuarantineStatus = Literal["quarantined", "released"]


@dataclass
class QuarantineRecord:
    document_id: str
    org_id: str
    reason: str | None
    status: QuarantineStatus
    quarantined_by: str
    quarantined_at: float
    released_by: str | None = None
    released_at: float | None = None


class QuarantineStore(ABC):
    @abstractmethod
    def quarantine(self, *, document_id: str, org_id: str, reason: str | None, actor: str) -> QuarantineRecord: ...

    @abstractmethod
    def release(self, *, document_id: str, org_id: str, actor: str) -> QuarantineRecord | None: ...

    @abstractmethod
    def list_quarantined(self, *, org_id: str) -> list[QuarantineRecord]: ...

    @abstractmethod
    def quarantined_ids(self, *, org_id: str) -> set[str]:
        """The document ids the Retriever must exclude for this org.

        Returns a set rather than a per-document `is_quarantined` check
        because retrieval filters a whole page of hits at once — one query
        per retrieve() call instead of one per hit.
        """


class InMemoryQuarantineStore(QuarantineStore):
    """Reference implementation, mirroring InMemoryReviewQueue — lets the
    whole system (and its tests) run without Postgres.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], QuarantineRecord] = {}

    def quarantine(self, *, document_id: str, org_id: str, reason: str | None, actor: str) -> QuarantineRecord:
        record = QuarantineRecord(
            document_id=document_id,
            org_id=org_id,
            reason=reason,
            status="quarantined",
            quarantined_by=actor,
            quarantined_at=time.time(),
        )
        self._records[(org_id, document_id)] = record
        return record

    def release(self, *, document_id: str, org_id: str, actor: str) -> QuarantineRecord | None:
        record = self._records.get((org_id, document_id))
        if record is None:
            return None
        record.status = "released"
        record.released_by = actor
        record.released_at = time.time()
        return record

    def list_quarantined(self, *, org_id: str) -> list[QuarantineRecord]:
        return [r for (o, _), r in self._records.items() if o == org_id and r.status == "quarantined"]

    def quarantined_ids(self, *, org_id: str) -> set[str]:
        return {r.document_id for r in self.list_quarantined(org_id=org_id)}


class PostgresQuarantineStore(QuarantineStore):
    """Persistent store backed by `quarantined_documents` (see db.py).

    Every statement carries an explicit `org_id` predicate even though RLS
    already scopes the table (security/rls.py) — defense in depth, and the
    same belt-and-braces the other Postgres stores here use.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def quarantine(self, *, document_id: str, org_id: str, reason: str | None, actor: str) -> QuarantineRecord:
        now = time.time()
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(quarantined_documents_table).where(
                    quarantined_documents_table.c.document_id == document_id,
                    quarantined_documents_table.c.org_id == org_id,
                )
            ).first()
            values = {
                "status": "quarantined",
                "reason": reason,
                "quarantined_by": actor,
                "quarantined_at": now,
                # Cleared so a re-quarantine after a release doesn't keep
                # stale release metadata on the row.
                "released_by": None,
                "released_at": None,
            }
            if existing is None:
                conn.execute(
                    insert(quarantined_documents_table).values(document_id=document_id, org_id=org_id, **values)
                )
            else:
                conn.execute(
                    update(quarantined_documents_table)
                    .where(
                        quarantined_documents_table.c.document_id == document_id,
                        quarantined_documents_table.c.org_id == org_id,
                    )
                    .values(**values)
                )
        return QuarantineRecord(
            document_id=document_id,
            org_id=org_id,
            reason=reason,
            status="quarantined",
            quarantined_by=actor,
            quarantined_at=now,
        )

    def release(self, *, document_id: str, org_id: str, actor: str) -> QuarantineRecord | None:
        now = time.time()
        with self._engine.begin() as conn:
            result = conn.execute(
                update(quarantined_documents_table)
                .where(
                    quarantined_documents_table.c.document_id == document_id,
                    quarantined_documents_table.c.org_id == org_id,
                )
                .values(status="released", released_by=actor, released_at=now)
            )
            if result.rowcount == 0:
                return None
            row = conn.execute(
                select(quarantined_documents_table).where(
                    quarantined_documents_table.c.document_id == document_id,
                    quarantined_documents_table.c.org_id == org_id,
                )
            ).one()
        return _to_record(row)

    def list_quarantined(self, *, org_id: str) -> list[QuarantineRecord]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(quarantined_documents_table)
                .where(
                    quarantined_documents_table.c.org_id == org_id,
                    quarantined_documents_table.c.status == "quarantined",
                )
                .order_by(quarantined_documents_table.c.quarantined_at.desc())
            ).all()
        return [_to_record(row) for row in rows]

    def quarantined_ids(self, *, org_id: str) -> set[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(quarantined_documents_table.c.document_id).where(
                    quarantined_documents_table.c.org_id == org_id,
                    quarantined_documents_table.c.status == "quarantined",
                )
            ).all()
        return {row.document_id for row in rows}


def _to_record(row) -> QuarantineRecord:
    return QuarantineRecord(
        document_id=row.document_id,
        org_id=row.org_id,
        reason=row.reason,
        status=row.status,
        quarantined_by=row.quarantined_by,
        quarantined_at=row.quarantined_at,
        released_by=row.released_by,
        released_at=row.released_at,
    )
