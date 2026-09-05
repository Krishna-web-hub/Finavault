"""Tests for the Cloud KMS KeyProvider.

No GCP credentials or network: `GcpKmsKeyProvider` takes an injectable
client, so what's under test here is this codebase's logic — request shape,
checksum verification, DEK cache behaviour, and the deliberate absence of a
fallback in `build_key_provider` — not Google's service.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pytest
from cryptography.exceptions import InvalidTag

from finvault.errors import ConfigurationError
from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider, chunk_aad
from finvault.security.kms import (
    GcpKmsKeyProvider,
    KmsIntegrityError,
    _crc32c,
    build_key_provider,
)

_KEY_NAME = "projects/p/locations/l/keyRings/r/cryptoKeys/k"


@dataclass
class _EncryptResponse:
    ciphertext: bytes
    verified_plaintext_crc32c: bool = True


@dataclass
class _DecryptResponse:
    plaintext: bytes
    plaintext_crc32c: int | None = None


class FakeKmsClient:
    """Reversible stand-in for Cloud KMS: wraps by prefixing a marker, so a
    round trip is verifiable without real crypto or a network call.
    """

    _PREFIX = b"kms::"

    def __init__(self) -> None:
        self.encrypt_calls: list[dict] = []
        self.decrypt_calls: list[dict] = []

    def encrypt(self, *, request: dict) -> _EncryptResponse:
        self.encrypt_calls.append(request)
        return _EncryptResponse(ciphertext=self._PREFIX + request["plaintext"])

    def decrypt(self, *, request: dict) -> _DecryptResponse:
        self.decrypt_calls.append(request)
        plaintext = request["ciphertext"][len(self._PREFIX) :]
        return _DecryptResponse(plaintext=plaintext, plaintext_crc32c=_crc32c(plaintext))


def _provider(client: FakeKmsClient, **kwargs) -> GcpKmsKeyProvider:
    kwargs.setdefault("cache_max_entries", 0)  # caching off unless a test asks
    kwargs.setdefault("cache_ttl_seconds", 0)
    return GcpKmsKeyProvider(_KEY_NAME, client=client, **kwargs)


# --- wrap / unwrap ----------------------------------------------------------


def test_wrap_and_unwrap_round_trip() -> None:
    client = FakeKmsClient()
    provider = _provider(client)
    dek = os.urandom(32)

    wrapped = provider.wrap_key(dek)

    assert wrapped != dek, "the DEK must not come back in the clear"
    assert provider.unwrap_key(wrapped) == dek


def test_requests_name_the_configured_key() -> None:
    client = FakeKmsClient()
    provider = _provider(client)

    provider.unwrap_key(provider.wrap_key(os.urandom(32)))

    assert client.encrypt_calls[0]["name"] == _KEY_NAME
    assert client.decrypt_calls[0]["name"] == _KEY_NAME


def test_integrity_checksums_are_sent_when_available() -> None:
    client = FakeKmsClient()
    provider = _provider(client)
    dek = os.urandom(32)

    wrapped = provider.wrap_key(dek)
    provider.unwrap_key(wrapped)

    if _crc32c(b"probe") is None:
        pytest.skip("google-crc32c not installed — checksums degrade to absent by design")
    assert client.encrypt_calls[0]["plaintext_crc32c"] == _crc32c(dek)
    assert client.decrypt_calls[0]["ciphertext_crc32c"] == _crc32c(wrapped)


def test_unverified_wrap_checksum_raises() -> None:
    """KMS reporting it could not verify the plaintext checksum means the
    payload may have been corrupted in transit — wrapping the wrong bytes
    would surface much later as an undecryptable chunk.
    """
    if _crc32c(b"probe") is None:
        pytest.skip("google-crc32c not installed")

    class Unverifying(FakeKmsClient):
        def encrypt(self, *, request: dict) -> _EncryptResponse:
            return _EncryptResponse(ciphertext=b"x", verified_plaintext_crc32c=False)

    with pytest.raises(KmsIntegrityError):
        _provider(Unverifying()).wrap_key(os.urandom(32))


def test_corrupted_unwrap_response_raises() -> None:
    class Corrupting(FakeKmsClient):
        def decrypt(self, *, request: dict) -> _DecryptResponse:
            # Checksum of different bytes than the plaintext returned.
            return _DecryptResponse(plaintext=b"a" * 32, plaintext_crc32c=_crc32c(b"b" * 32))

    if _crc32c(b"probe") is None:
        pytest.skip("google-crc32c not installed")

    with pytest.raises(KmsIntegrityError):
        _provider(Corrupting()).unwrap_key(b"kms::whatever")


# --- the DEK cache ----------------------------------------------------------


def test_unwrap_is_cached_but_wrap_is_not() -> None:
    client = FakeKmsClient()
    provider = _provider(client, cache_max_entries=16, cache_ttl_seconds=60)

    first = provider.wrap_key(os.urandom(32))
    second = provider.wrap_key(os.urandom(32))
    assert len(client.encrypt_calls) == 2, "every wrap is a fresh DEK — nothing to cache"

    provider.unwrap_key(first)
    provider.unwrap_key(first)
    provider.unwrap_key(first)
    assert len(client.decrypt_calls) == 1, "repeat unwraps of one chunk must not re-hit KMS"

    provider.unwrap_key(second)
    assert len(client.decrypt_calls) == 2, "a different chunk still goes to KMS"


def test_cache_entries_expire() -> None:
    client = FakeKmsClient()
    provider = _provider(client, cache_max_entries=16, cache_ttl_seconds=0.05)
    wrapped = provider.wrap_key(os.urandom(32))

    provider.unwrap_key(wrapped)
    time.sleep(0.08)
    provider.unwrap_key(wrapped)

    assert len(client.decrypt_calls) == 2, "an expired entry must go back to KMS"


def test_cache_is_bounded() -> None:
    client = FakeKmsClient()
    provider = _provider(client, cache_max_entries=2, cache_ttl_seconds=60)
    wrapped = [provider.wrap_key(os.urandom(32)) for _ in range(3)]

    for w in wrapped:
        provider.unwrap_key(w)
    assert len(client.decrypt_calls) == 3

    # The first entry was evicted by the third insert, so it re-hits KMS
    # while the most recent one is still served from cache.
    provider.unwrap_key(wrapped[0])
    assert len(client.decrypt_calls) == 4
    provider.unwrap_key(wrapped[2])
    assert len(client.decrypt_calls) == 4


def test_invalidate_cache_forces_kms_on_the_next_unwrap() -> None:
    """The escape hatch for revocation: without it, a revoked IAM binding
    would not take effect until every cached entry aged out.
    """
    client = FakeKmsClient()
    provider = _provider(client, cache_max_entries=16, cache_ttl_seconds=600)
    wrapped = provider.wrap_key(os.urandom(32))
    provider.unwrap_key(wrapped)

    provider.invalidate_cache()
    provider.unwrap_key(wrapped)

    assert len(client.decrypt_calls) == 2


def test_caching_disabled_sends_every_unwrap_to_kms() -> None:
    client = FakeKmsClient()
    provider = _provider(client, cache_max_entries=0, cache_ttl_seconds=0)
    wrapped = provider.wrap_key(os.urandom(32))

    provider.unwrap_key(wrapped)
    provider.unwrap_key(wrapped)

    assert len(client.decrypt_calls) == 2


# --- it actually works as a KeyProvider -------------------------------------


def test_envelope_encryptor_round_trips_through_kms() -> None:
    """The provider is only useful if EnvelopeEncryptor can use it unchanged
    — the whole point of the KeyProvider seam.
    """
    encryptor = EnvelopeEncryptor(_provider(FakeKmsClient(), cache_max_entries=16, cache_ttl_seconds=60))
    aad = chunk_aad("doc-1", 0)

    payload = encryptor.encrypt("Q3 revenue was $10 million.", aad=aad)

    assert b"Q3 revenue" not in payload.ciphertext
    assert encryptor.decrypt(payload, aad=aad) == "Q3 revenue was $10 million."


def test_mismatched_aad_still_fails_under_kms() -> None:
    """Chunk binding is a property of EnvelopeEncryptor, not the provider —
    swapping in KMS must not weaken it.
    """
    encryptor = EnvelopeEncryptor(_provider(FakeKmsClient()))
    payload = encryptor.encrypt("secret", aad=chunk_aad("doc-1", 0))

    with pytest.raises(InvalidTag):
        encryptor.decrypt(payload, aad=chunk_aad("doc-1", 1))


def test_empty_key_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        GcpKmsKeyProvider("", client=FakeKmsClient())


# --- build_key_provider -----------------------------------------------------


def test_build_key_provider_defaults_to_local(monkeypatch, tmp_path) -> None:
    from finvault.config import settings

    monkeypatch.setattr(settings, "finvault_key_provider", "local")
    monkeypatch.setattr(settings, "finvault_master_key_path", tmp_path / "master.key")

    assert isinstance(build_key_provider(), LocalKeyProvider)


def test_gcp_kms_without_a_key_name_fails_startup(monkeypatch) -> None:
    """Deliberately not a fallback to local: silently switching KEK identity
    would make every existing chunk undecryptable and protect new ones with
    weaker custody than the operator asked for.
    """
    from finvault.config import settings

    monkeypatch.setattr(settings, "finvault_key_provider", "gcp_kms")
    monkeypatch.setattr(settings, "finvault_gcp_kms_key_name", None)

    with pytest.raises(ConfigurationError):
        build_key_provider()


def test_unknown_key_provider_fails_startup(monkeypatch) -> None:
    from finvault.config import settings

    monkeypatch.setattr(settings, "finvault_key_provider", "azure_vault")

    with pytest.raises(ConfigurationError):
        build_key_provider()
