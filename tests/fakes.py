"""Minimal fakes for the OpenAI chat-completions surface, used to test the
agent tool-loop and compliance-review logic without hitting a live API or
spending real credits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from finvault.agents.analyst_agent import AnalystAnswer
from finvault.ingestion.embeddings import EmbeddingProvider
from finvault.models import Classification


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding: hashes each text into a
    small fixed-size vector. Not semantically meaningful — good enough to
    construct a real Retriever in tests without loading a real model.
    """

    _DIM = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self._DIM]]

    @property
    def dimension(self) -> int:
        return self._DIM

    @property
    def name(self) -> str:
        return "fake-hash-embedding"


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunctionCall
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[Any] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None or not exclude_none:
            data["content"] = self.content
        if self.tool_calls is not None or not exclude_none:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    total_tokens: int


@dataclass
class FakeResponse:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None

    @classmethod
    def text(cls, content: str, *, finish_reason: str = "stop", total_tokens: int = 50) -> FakeResponse:
        return cls(
            choices=[FakeChoice(message=FakeMessage(content=content), finish_reason=finish_reason)],
            usage=FakeUsage(total_tokens=total_tokens),
        )

    @classmethod
    def tool_call(
        cls, name: str, arguments: dict[str, Any], *, call_id: str = "call_1", total_tokens: int = 50
    ) -> FakeResponse:
        import json as _json

        tool_call = FakeToolCall(id=call_id, function=FakeFunctionCall(name=name, arguments=_json.dumps(arguments)))
        return cls(
            choices=[FakeChoice(message=FakeMessage(content=None, tool_calls=[tool_call]), finish_reason="tool_calls")],
            usage=FakeUsage(total_tokens=total_tokens),
        )


def _fake_request() -> Any:
    import httpx

    return httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def make_rate_limit_error(message: str = "rate limited") -> Exception:
    """A real openai.RateLimitError — retryable (see agents/base.py)."""
    import httpx
    from openai import RateLimitError

    response = httpx.Response(status_code=429, request=_fake_request())
    return RateLimitError(message, response=response, body=None)


def make_connection_error(message: str = "connection reset") -> Exception:
    """A real openai.APIConnectionError — retryable (see agents/base.py)."""
    from openai import APIConnectionError

    return APIConnectionError(message=message, request=_fake_request())


def make_bad_request_error(message: str = "invalid request") -> Exception:
    """A real openai.BadRequestError — NOT retryable (see agents/base.py)."""
    import httpx
    from openai import BadRequestError

    response = httpx.Response(status_code=400, request=_fake_request())
    return BadRequestError(message, response=response, body=None)


class FakeCompletions:
    """Pops one scripted response (or raises a scripted exception) per call."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        # Callers (Agent.run) pass a `messages` list they keep mutating
        # in place across loop iterations — store a shallow copy so each
        # call's snapshot reflects the request as it was at call time, not
        # whatever the list looks like by the time the test inspects it.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
        if not self._responses:
            raise AssertionError("FakeCompletions ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.chat = FakeChat(FakeCompletions(responses))


class FakeRetrieverAgent:
    """Stands in for RetrieverAgent so a test can exercise Orchestrator
    wiring without a real vector store/encryption pipeline.

    Signature must match RetrieverAgent.run() exactly, including `budget` —
    a prior version of this fixture omitted it, which meant every test using
    it was silently calling search_documents against a TypeError (swallowed
    by base.py's generic tool-error handling and fed back to the model as an
    error string) without any test noticing. See orchestrator.py's
    retrieved_document_ids for why that property exists (Milestone 4's
    query-scoped graph_data).
    """

    def __init__(
        self,
        result: str = "Retrieved context: Q3 revenue was $10 million.",
        max_classification: Classification = Classification.INTERNAL,
        retrieved_document_ids: set[str] | None = None,
        injection_flags: list[dict] | None = None,
    ) -> None:
        self._result = result
        self.max_classification_seen = max_classification
        self.retrieved_document_ids = retrieved_document_ids or set()
        # Shape matches what the real RetrieverAgent appends per flagged
        # chunk (see agents/retriever_agent.py): document, chunk_id, patterns.
        self._injection_flags = injection_flags or []

    def run(self, query: str, *, budget=None) -> tuple[str, list[dict]]:
        return self._result, list(self._injection_flags)


class FakeAnalystAgent:
    """Stands in for AnalystAgent — always returns the same scripted answer
    with no citations, regardless of the question/context it's given.
    """

    def __init__(self, *, response: str) -> None:
        self._response = response

    def run_structured(self, prompt: str, *, budget=None) -> AnalystAnswer:
        return AnalystAnswer(answer=self._response, citations=[])
