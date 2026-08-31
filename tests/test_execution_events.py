from __future__ import annotations

from finvault.agents.canvas_models import ExecutionStepNode
from finvault.agents.execution_events import ExecutionEvent, ExecutionEventBus


def _make_step(**overrides) -> ExecutionStepNode:
    defaults = {
        "step_id": "s1",
        "agent_name": "retriever",
        "action": "search_documents",
        "status": "success",
        "duration_ms": 12.5,
        "payload_preview": "preview",
        "timestamp": 0.0,
    }
    defaults.update(overrides)
    return ExecutionStepNode(**defaults)


def test_publish_appends_to_events_in_order() -> None:
    bus = ExecutionEventBus()
    bus.publish(ExecutionEvent(type="step_started", agent="retriever", action="search_documents"))
    bus.publish(ExecutionEvent(type="step_finished", agent="retriever", action="search_documents", step=_make_step()))

    assert [e.type for e in bus.events] == ["step_started", "step_finished"]


def test_events_property_returns_a_copy_not_the_live_list() -> None:
    bus = ExecutionEventBus()
    bus.publish(ExecutionEvent(type="step_started", agent="retriever", action="search_documents"))

    snapshot = bus.events
    snapshot.append(ExecutionEvent(type="step_started", agent="analyst", action="analyze"))

    assert len(bus.events) == 1  # mutating the snapshot must not affect the bus


def test_subscribers_are_notified_synchronously_as_events_are_published() -> None:
    bus = ExecutionEventBus()
    received: list[ExecutionEvent] = []
    bus.subscribe(received.append)

    event = ExecutionEvent(type="step_started", agent="retriever", action="search_documents")
    bus.publish(event)

    assert received == [event]


def test_multiple_subscribers_all_receive_the_same_event() -> None:
    bus = ExecutionEventBus()
    received_a: list[ExecutionEvent] = []
    received_b: list[ExecutionEvent] = []
    bus.subscribe(received_a.append)
    bus.subscribe(received_b.append)

    bus.publish(ExecutionEvent(type="step_started", agent="analyst", action="analyze"))

    assert len(received_a) == 1
    assert len(received_b) == 1


def test_step_finished_event_carries_the_completed_step() -> None:
    bus = ExecutionEventBus()
    received: list[ExecutionEvent] = []
    bus.subscribe(received.append)

    step = _make_step(status="vetoed", duration_ms=42.0)
    bus.publish(ExecutionEvent(type="step_finished", agent="compliance", action="review_output", step=step))

    assert received[0].step is step
    assert received[0].step.status == "vetoed"
