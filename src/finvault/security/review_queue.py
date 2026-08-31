"""Human-in-the-loop queue for compliance-blocked responses.

Today a block just returns a generic "requires manual handling" message and
writes an audit entry — nothing turns that into something a compliance
officer can act on. This queue is that surface: every block is enqueued
here with full context (the raw draft answer, not just the fact that
something was blocked), and a compliance officer can review and record a
decision.

Showing the raw, pre-redaction draft answer to reviewers is deliberate, not
an oversight: `ComplianceVerdict.redacted_answer` is empty on a block by
design (see agents/compliance_agent.py) so nothing leaks to the end user,
but a reviewer adjudicating *why* something was blocked needs to see what
was actually flagged. COMPLIANCE_OFFICER is already this system's highest
RBAC tier (see models.py) — the role that can read RESTRICTED-classified
documents — so this isn't a new trust boundary, just this role's existing
clearance applied to a new surface.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from finvault.db import review_queue_table
from finvault.errors import ReviewItemNotFoundError, ReviewQueueError

ReviewStatus = Literal["pending", "released", "denied"]


@dataclass
class ReviewItem:
    id: str
    org_id: str
    user_id: str
    question: str
    draft_answer: str
    block_reason: str | None
    findings: list[str]
    citations: list[dict]
    created_at: float
    status: ReviewStatus = "pending"
    reviewed_by: str | None = None
    reviewed_at: float | None = None
    reviewer_note: str | None = None


class ReviewQueue(ABC):
    @abstractmethod
    def enqueue(
        self,
        *,
        org_id: str,
        user_id: str,
        question: str,
        draft_answer: str,
        block_reason: str | None,
        findings: list[str],
        citations: list[dict],
    ) -> ReviewItem: ...

    @abstractmethod
    def list_pending(self, *, org_id: str) -> list[ReviewItem]: ...

    @abstractmethod
    def get(self, item_id: str) -> ReviewItem | None: ...

    @abstractmethod
    def resolve(
        self, item_id: str, *, org_id: str, status: ReviewStatus, reviewed_by: str, reviewer_note: str | None = None
    ) -> ReviewItem: ...


class InMemoryReviewQueue(ReviewQueue):
    """Reference implementation — not persisted across process restarts. A
    production deployment would back this with Postgres, the same pattern
    already used for AuditLog (security/audit.py).
    """

    def __init__(self) -> None:
        self._items: dict[str, ReviewItem] = {}

    def enqueue(
        self,
        *,
        org_id: str,
        user_id: str,
        question: str,
        draft_answer: str,
        block_reason: str | None,
        findings: list[str],
        citations: list[dict],
    ) -> ReviewItem:
        item = ReviewItem(
            id=str(uuid.uuid4()),
            org_id=org_id,
            user_id=user_id,
            question=question,
            draft_answer=draft_answer,
            block_reason=block_reason,
            findings=findings,
            citations=citations,
            created_at=time.time(),
        )
        self._items[item.id] = item
        return item

    def list_pending(self, *, org_id: str) -> list[ReviewItem]:
        # Org-scoped: a compliance officer only ever sees their own org's
        # blocked items, mirroring the org isolation already enforced
        # throughout retrieval (retrieval/retriever.py).
        return [item for item in self._items.values() if item.org_id == org_id and item.status == "pending"]

    def get(self, item_id: str) -> ReviewItem | None:
        return self._items.get(item_id)

    def resolve(
        self, item_id: str, *, org_id: str, status: ReviewStatus, reviewed_by: str, reviewer_note: str | None = None
    ) -> ReviewItem:
        item = self._items.get(item_id)
        if item is None:
            raise ReviewItemNotFoundError(f"No review item with id '{item_id}'", context={"item_id": item_id})
        if item.org_id != org_id:
            # Same non-inference posture as everywhere else in this system:
            # a cross-org lookup fails the same way a missing item does,
            # rather than confirming the item exists in another org.
            raise ReviewItemNotFoundError(
                f"Review item '{item_id}' belongs to another org",
                context={"item_id": item_id, "requesting_org_id": org_id},
            )
        if status == "pending":
            raise ReviewQueueError(
                "Cannot resolve an item back to 'pending'", context={"item_id": item_id, "requested_status": status}
            )

        item.status = status
        item.reviewed_by = reviewed_by
        item.reviewed_at = time.time()
        item.reviewer_note = reviewer_note
        return item


class PostgresReviewQueue(ReviewQueue):
    """Persistent review queue backed by the `review_queue` table (see
    db.py). Survives process restarts and is shared correctly across
    multiple workers — the two problems InMemoryReviewQueue has in a real
    deployment.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def enqueue(
        self,
        *,
        org_id: str,
        user_id: str,
        question: str,
        draft_answer: str,
        block_reason: str | None,
        findings: list[str],
        citations: list[dict],
    ) -> ReviewItem:
        item = ReviewItem(
            id=str(uuid.uuid4()),
            org_id=org_id,
            user_id=user_id,
            question=question,
            draft_answer=draft_answer,
            block_reason=block_reason,
            findings=findings,
            citations=citations,
            created_at=time.time(),
        )
        with self._engine.begin() as conn:
            conn.execute(
                insert(review_queue_table).values(
                    id=item.id,
                    org_id=item.org_id,
                    user_id=item.user_id,
                    question=item.question,
                    draft_answer=item.draft_answer,
                    block_reason=item.block_reason,
                    findings=item.findings,
                    citations=item.citations,
                    status=item.status,
                    created_at=item.created_at,
                    reviewed_by=item.reviewed_by,
                    reviewed_at=item.reviewed_at,
                    reviewer_note=item.reviewer_note,
                )
            )
        return item

    def list_pending(self, *, org_id: str) -> list[ReviewItem]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(review_queue_table).where(
                    review_queue_table.c.org_id == org_id, review_queue_table.c.status == "pending"
                )
            ).all()
        return [self._row_to_item(row) for row in rows]

    def get(self, item_id: str) -> ReviewItem | None:
        with self._engine.connect() as conn:
            row = conn.execute(select(review_queue_table).where(review_queue_table.c.id == item_id)).first()
        return self._row_to_item(row) if row is not None else None

    def resolve(
        self, item_id: str, *, org_id: str, status: ReviewStatus, reviewed_by: str, reviewer_note: str | None = None
    ) -> ReviewItem:
        if status == "pending":
            raise ReviewQueueError(
                "Cannot resolve an item back to 'pending'", context={"item_id": item_id, "requested_status": status}
            )

        reviewed_at = time.time()
        with self._engine.begin() as conn:
            row = conn.execute(select(review_queue_table).where(review_queue_table.c.id == item_id)).first()
            if row is None or row.org_id != org_id:
                # Same non-inference posture as the in-memory version: a
                # cross-org lookup fails the same way a missing item does.
                raise ReviewItemNotFoundError(
                    f"No review item with id '{item_id}' in org '{org_id}'",
                    context={"item_id": item_id, "requesting_org_id": org_id},
                )
            conn.execute(
                update(review_queue_table)
                .where(review_queue_table.c.id == item_id)
                .values(status=status, reviewed_by=reviewed_by, reviewed_at=reviewed_at, reviewer_note=reviewer_note)
            )
        return ReviewItem(
            id=row.id,
            org_id=row.org_id,
            user_id=row.user_id,
            question=row.question,
            draft_answer=row.draft_answer,
            block_reason=row.block_reason,
            findings=row.findings,
            citations=row.citations,
            created_at=row.created_at,
            status=status,
            reviewed_by=reviewed_by,
            reviewed_at=reviewed_at,
            reviewer_note=reviewer_note,
        )

    def _row_to_item(self, row: Any) -> ReviewItem:
        return ReviewItem(
            id=row.id,
            org_id=row.org_id,
            user_id=row.user_id,
            question=row.question,
            draft_answer=row.draft_answer,
            block_reason=row.block_reason,
            findings=row.findings,
            citations=row.citations,
            created_at=row.created_at,
            status=row.status,
            reviewed_by=row.reviewed_by,
            reviewed_at=row.reviewed_at,
            reviewer_note=row.reviewer_note,
        )
