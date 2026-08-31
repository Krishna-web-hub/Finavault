"""Hash-chained, append-only audit log.

Every query, retrieval, ACL decision, and guardrail action is recorded here.
Each entry's hash is derived from its content plus the previous entry's hash,
so `verify_chain()` can detect any out-of-band tampering with the log —
insertion, deletion, or edit of a past entry breaks the chain from that point
forward.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from finvault.db import audit_log_table

GENESIS_HASH = "0" * 64


def _compute_hash(
    timestamp: float, actor: str, action: str, resource: str, details: dict[str, Any], prev_hash: str
) -> str:
    payload = json.dumps(
        {
            "timestamp": timestamp,
            "actor": actor,
            "action": action,
            "resource": resource,
            "details": details,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp: float
    actor: str
    action: str
    resource: str
    details: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class AuditLog(ABC):
    @abstractmethod
    def append(
        self, *, actor: str, action: str, resource: str, details: dict[str, Any] | None = None
    ) -> AuditEntry: ...

    @abstractmethod
    def entries(self) -> list[AuditEntry]: ...

    def verify_chain(self) -> bool:
        """Recomputes every entry's hash from its content and checks the chain
        links prev_hash -> entry_hash consistently from the genesis hash.
        """
        prev = GENESIS_HASH
        for entry in self.entries():
            if entry.prev_hash != prev:
                return False
            expected = _compute_hash(
                entry.timestamp, entry.actor, entry.action, entry.resource, entry.details, entry.prev_hash
            )
            if expected != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True


class InMemoryAuditLog(AuditLog):
    """Reference implementation — used by default when no Postgres engine is
    configured (e.g. unit tests, quick local scripts). Not persisted across
    process restarts.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(self, *, actor: str, action: str, resource: str, details: dict[str, Any] | None = None) -> AuditEntry:
        details = details or {}
        prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        timestamp = time.time()
        entry_hash = _compute_hash(timestamp, actor, action, resource, details, prev_hash)
        entry = AuditEntry(
            seq=len(self._entries),
            timestamp=timestamp,
            actor=actor,
            action=action,
            resource=resource,
            details=details,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)


class PostgresAuditLog(AuditLog):
    """Persistent audit log backed by the `audit_log` table (see db.py).

    Note: append() reads the last row's hash and inserts within one
    transaction, which is correct for a single-writer or low-concurrency
    deployment. A high-throughput multi-writer deployment should serialize
    appends through a dedicated writer process or a DB-level advisory lock to
    avoid a race between two concurrent appends computing the same prev_hash.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, *, actor: str, action: str, resource: str, details: dict[str, Any] | None = None) -> AuditEntry:
        details = details or {}
        with self._engine.begin() as conn:
            last = conn.execute(
                select(audit_log_table.c.entry_hash).order_by(audit_log_table.c.seq.desc()).limit(1)
            ).first()
            prev_hash = last.entry_hash if last else GENESIS_HASH
            timestamp = time.time()
            entry_hash = _compute_hash(timestamp, actor, action, resource, details, prev_hash)
            result = conn.execute(
                insert(audit_log_table).values(
                    timestamp=timestamp,
                    actor=actor,
                    action=action,
                    resource=resource,
                    details=details,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                )
            )
            seq = result.inserted_primary_key[0]
        return AuditEntry(
            seq=seq,
            timestamp=timestamp,
            actor=actor,
            action=action,
            resource=resource,
            details=details,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    def entries(self) -> list[AuditEntry]:
        with self._engine.connect() as conn:
            rows = conn.execute(select(audit_log_table).order_by(audit_log_table.c.seq.asc())).all()
        return [
            AuditEntry(
                seq=row.seq,
                timestamp=row.timestamp,
                actor=row.actor,
                action=row.action,
                resource=row.resource,
                details=row.details,
                prev_hash=row.prev_hash,
                entry_hash=row.entry_hash,
            )
            for row in rows
        ]
