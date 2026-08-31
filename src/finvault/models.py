"""Shared domain schemas.

Classification and Role are the two axes access control is built on
(see security/access_control.py): every Document/Chunk carries a
classification, every User carries a role and an org_id, and every retrieval
is checked against both before any plaintext is produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Classification(StrEnum):
    """Data sensitivity tier. Travels with every document/chunk and gates both
    who can retrieve it (access_control.py) and whether it may ever be
    included in a prompt sent to the LLM (guardrails.enforce_externalization_policy).
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Role(StrEnum):
    """User clearance level, ordered by ROLE_RANK below."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"


ROLE_RANK: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.ANALYST: 1,
    Role.COMPLIANCE_OFFICER: 2,
    Role.ADMIN: 3,
}

# Minimum role required to retrieve content at each classification tier.
CLASSIFICATION_MIN_ROLE: dict[Classification, Role] = {
    Classification.PUBLIC: Role.VIEWER,
    Classification.INTERNAL: Role.VIEWER,
    Classification.CONFIDENTIAL: Role.ANALYST,
    Classification.RESTRICTED: Role.COMPLIANCE_OFFICER,
}

# Ordering for "highest classification seen this turn" tracking — used to
# decide, once per request, whether anything retrieved was sensitive enough
# to block externalization to the LLM (see security/guardrails.py).
CLASSIFICATION_RANK: dict[Classification, int] = {
    Classification.PUBLIC: 0,
    Classification.INTERNAL: 1,
    Classification.CONFIDENTIAL: 2,
    Classification.RESTRICTED: 3,
}


class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    role: Role
    org_id: str


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    title: str
    source_path: str | None = None
    classification: Classification
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """Metadata for one chunk. Deliberately has no plaintext field — plaintext
    only ever exists transiently in memory during ingestion (before
    encryption) and retrieval (after decryption + ACL check); it is never a
    persisted attribute of this model.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    org_id: str
    classification: Classification
    chunk_index: int


class Query(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    user_id: str
    org_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievedChunk(BaseModel):
    """A chunk after it has passed the ACL check and been decrypted — the only
    place plaintext chunk content exists as a named field, and only ever
    in-memory, never persisted.
    """

    chunk: Chunk
    document_title: str
    text: str
    score: float


class RetrievedDocument(BaseModel):
    """A whole document's plaintext, reassembled in chunk_index order after
    every one of its chunks has passed the ACL check — see
    Retriever.get_document_text(). Used for whole-document operations (e.g.
    comparison, agents/comparison_agent.py) where a handful of retrieved
    chunks isn't enough context; never persisted, same as RetrievedChunk.
    """

    document_id: str
    title: str
    text: str
