"""Live execution-event pub/sub for the Execution DAG canvas.

The bus exists and Orchestrator publishes to it at each real checkpoint
(see orchestrator.py's `_run_step`), with subscribers notified
synchronously as events happen. The SSE route (api/routes.py's
`POST /query/stream`) consumes it: a background thread runs the
synchronous `Orchestrator.handle()` call, and events cross into the
async generator through a thread-safe `queue.Queue`, drained via
`run_in_executor` so the event loop isn't blocked.

One bus instance per in-flight request (see Orchestrator.handle(), which
creates a fresh one every call) — never a shared/global bus, the same way
TokenBudget is per-request rather than per-Orchestrator-instance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from finvault.agents.canvas_models import ExecutionStepNode

EventType = Literal["step_started", "step_finished"]


@dataclass(frozen=True)
class ExecutionEvent:
    type: EventType
    agent: str
    action: str
    # Populated on "step_finished" only — a "step_started" event announces
    # that a step began, before there's a completed ExecutionStepNode
    # (with its real measured duration) to attach.
    step: ExecutionStepNode | None = None
    timestamp: float = field(default_factory=time.time)


Subscriber = Callable[[ExecutionEvent], None]


class ExecutionEventBus:
    def __init__(self) -> None:
        self._events: list[ExecutionEvent] = []
        self._subscribers: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    def publish(self, event: ExecutionEvent) -> None:
        self._events.append(event)
        for callback in self._subscribers:
            callback(event)

    @property
    def events(self) -> list[ExecutionEvent]:
        return list(self._events)
