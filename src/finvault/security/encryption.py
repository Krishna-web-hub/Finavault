"""Envelope encryption for chunk content.

Each chunk gets its own AES-256-GCM data key (DEK). The DEK is wrapped by a
key-encryption-key (KEK) held by a KeyProvider. A compromised vector store
yields ciphertext and wrapped keys — not plaintext, and not the KEK needed
to unwrap them.

KeyProvider is the swap point for production: LocalKeyProvider (file-backed)
is the dev/self-hosted default. An AWS KMS or HashiCorp Vault-backed
implementation of the same interface drops in with no changes anywhere else
in the codebase.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class KeyProvider(ABC):
    """Supplies and unwraps the key-encryption-key (KEK)."""

    @abstractmethod
    def wrap_key(self, data_key: bytes) -> bytes: ...

    @abstractmethod
    def unwrap_key(self, wrapped_key: bytes) -> bytes: ...


class LocalKeyProvider(KeyProvider):
    """Dev/self-hosted default: the KEK is a 256-bit key stored in a local
    file, created with owner-only permissions on first use.

    This is the *interface's* reference implementation, not a substitute for
    a real KMS. Swap for a KMS/Vault-backed KeyProvider in any deployment
    where the host filesystem isn't itself a trusted boundary.
    """

    def __init__(self, key_path: Path) -> None:
        self._key_path = key_path
        self._kek = self._load_or_create_kek()

    def _load_or_create_kek(self) -> bytes:
        if self._key_path.exists():
            return base64.urlsafe_b64decode(self._key_path.read_bytes())
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        kek = AESGCM.generate_key(bit_length=256)
        self._key_path.write_bytes(base64.urlsafe_b64encode(kek))
        self._key_path.chmod(0o600)
        return kek

    def wrap_key(self, data_key: bytes) -> bytes:
        aesgcm = AESGCM(self._kek)
        nonce = os.urandom(12)
        return nonce + aesgcm.encrypt(nonce, data_key, None)

    def unwrap_key(self, wrapped_key: bytes) -> bytes:
        aesgcm = AESGCM(self._kek)
        nonce, ciphertext = wrapped_key[:12], wrapped_key[12:]
        return aesgcm.decrypt(nonce, ciphertext, None)


@dataclass(frozen=True)
class EncryptedPayload:
    ciphertext: bytes
    nonce: bytes
    wrapped_dek: bytes

    def to_dict(self) -> dict[str, str]:
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "wrapped_dek": base64.b64encode(self.wrapped_dek).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> EncryptedPayload:
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            wrapped_dek=base64.b64decode(data["wrapped_dek"]),
        )


class EnvelopeEncryptor:
    """Per-chunk AES-256-GCM envelope encryption on top of a KeyProvider."""

    def __init__(self, key_provider: KeyProvider) -> None:
        self._key_provider = key_provider

    def encrypt(self, plaintext: str, *, aad: bytes | None = None) -> EncryptedPayload:
        """`aad` should bind the ciphertext to its context (e.g. document_id +
        chunk_index) so ciphertext from one chunk can't be silently swapped
        for another's during decryption — GCM will reject a mismatched aad.
        """
        dek = AESGCM.generate_key(bit_length=256)
        aesgcm = AESGCM(dek)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
        wrapped_dek = self._key_provider.wrap_key(dek)
        return EncryptedPayload(ciphertext=ciphertext, nonce=nonce, wrapped_dek=wrapped_dek)

    def decrypt(self, payload: EncryptedPayload, *, aad: bytes | None = None) -> str:
        dek = self._key_provider.unwrap_key(payload.wrapped_dek)
        aesgcm = AESGCM(dek)
        plaintext = aesgcm.decrypt(payload.nonce, payload.ciphertext, aad)
        return plaintext.decode("utf-8")


def chunk_aad(document_id: str, chunk_index: int) -> bytes:
    """Deterministic AAD binding a chunk's ciphertext to its identity."""
    return f"{document_id}:{chunk_index}".encode()
