from __future__ import annotations

import dataclasses

from finvault.security.audit import InMemoryAuditLog


def test_audit_chain_verifies_when_untampered() -> None:
    log = InMemoryAuditLog()
    log.append(actor="user-1", action="query", resource="orchestrator", details={"q": "test"})
    log.append(actor="user-1", action="retrieve", resource="vector_store", details={"returned": 3})
    assert log.verify_chain() is True


def test_audit_chain_detects_tampering() -> None:
    log = InMemoryAuditLog()
    log.append(actor="user-1", action="query", resource="orchestrator", details={"q": "test"})
    entry = log.append(actor="user-1", action="retrieve", resource="vector_store", details={"returned": 3})

    # Simulate out-of-band tampering with a past entry's content (the hash
    # itself is left as originally computed, as an attacker who can only
    # edit stored rows would do).
    tampered = dataclasses.replace(entry, details={"returned": 999})
    log._entries[-1] = tampered  # type: ignore[attr-defined]

    assert log.verify_chain() is False
