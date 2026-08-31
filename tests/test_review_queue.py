from __future__ import annotations

import pytest

from finvault.security.review_queue import InMemoryReviewQueue, ReviewQueueError


def _enqueue(queue: InMemoryReviewQueue, *, org_id: str = "org-a"):
    return queue.enqueue(
        org_id=org_id,
        user_id="user-1",
        question="What was net income?",
        draft_answer="Net income was $50 million.",
        block_reason="citation verification failed",
        findings=[],
        citations=[{"document": "Q1 Report", "quoted_text": "Net income was $50 million"}],
    )


def test_enqueue_then_list_pending() -> None:
    queue = InMemoryReviewQueue()
    item = _enqueue(queue)

    pending = queue.list_pending(org_id="org-a")

    assert len(pending) == 1
    assert pending[0].id == item.id
    assert pending[0].status == "pending"


def test_list_pending_is_org_scoped() -> None:
    queue = InMemoryReviewQueue()
    _enqueue(queue, org_id="org-a")
    _enqueue(queue, org_id="org-b")

    assert len(queue.list_pending(org_id="org-a")) == 1
    assert len(queue.list_pending(org_id="org-b")) == 1


def test_resolve_released_removes_item_from_pending_list() -> None:
    queue = InMemoryReviewQueue()
    item = _enqueue(queue)

    resolved = queue.resolve(item.id, org_id="org-a", status="released", reviewed_by="officer-1", reviewer_note="ok")

    assert resolved.status == "released"
    assert resolved.reviewed_by == "officer-1"
    assert resolved.reviewer_note == "ok"
    assert queue.list_pending(org_id="org-a") == []


def test_resolve_denied_is_recorded() -> None:
    queue = InMemoryReviewQueue()
    item = _enqueue(queue)

    resolved = queue.resolve(item.id, org_id="org-a", status="denied", reviewed_by="officer-1")

    assert resolved.status == "denied"
    assert queue.get(item.id).status == "denied"


def test_resolve_unknown_item_raises() -> None:
    queue = InMemoryReviewQueue()
    with pytest.raises(ReviewQueueError):
        queue.resolve("does-not-exist", org_id="org-a", status="released", reviewed_by="officer-1")


def test_resolve_fails_for_wrong_org_same_as_missing_item() -> None:
    """Cross-org access must fail the same way a missing item does — not
    confirm the item exists in another org (non-inference, same posture as
    everywhere else in this system).
    """
    queue = InMemoryReviewQueue()
    item = _enqueue(queue, org_id="org-a")

    with pytest.raises(ReviewQueueError):
        queue.resolve(item.id, org_id="org-b", status="released", reviewed_by="officer-1")


def test_resolve_to_pending_is_rejected() -> None:
    queue = InMemoryReviewQueue()
    item = _enqueue(queue)
    with pytest.raises(ReviewQueueError):
        queue.resolve(item.id, org_id="org-a", status="pending", reviewed_by="officer-1")
