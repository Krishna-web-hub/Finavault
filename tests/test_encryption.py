from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from finvault.security.encryption import EnvelopeEncryptor, LocalKeyProvider, chunk_aad


def _encryptor(tmp_path):
    return EnvelopeEncryptor(LocalKeyProvider(tmp_path / "master.key"))


def test_round_trip(tmp_path):
    enc = _encryptor(tmp_path)
    aad = chunk_aad("doc-1", 0)
    payload = enc.encrypt("some sensitive financial text", aad=aad)
    assert enc.decrypt(payload, aad=aad) == "some sensitive financial text"


def test_each_encryption_uses_a_fresh_data_key(tmp_path):
    enc = _encryptor(tmp_path)
    aad = chunk_aad("doc-1", 0)
    a = enc.encrypt("same text", aad=aad)
    b = enc.encrypt("same text", aad=aad)
    assert a.wrapped_dek != b.wrapped_dek
    assert a.ciphertext != b.ciphertext


def test_tampered_ciphertext_fails_closed(tmp_path):
    enc = _encryptor(tmp_path)
    aad = chunk_aad("doc-1", 0)
    payload = enc.encrypt("original text", aad=aad)
    flipped_last_byte = payload.ciphertext[:-1] + bytes([payload.ciphertext[-1] ^ 0xFF])
    tampered = type(payload)(ciphertext=flipped_last_byte, nonce=payload.nonce, wrapped_dek=payload.wrapped_dek)
    # InvalidTag specifically, not any Exception: the point of this test is
    # that AEAD *authentication* rejected the ciphertext. A different
    # exception would also pass a blind `raises(Exception)` while meaning
    # something entirely different went wrong.
    with pytest.raises(InvalidTag):
        enc.decrypt(tampered, aad=aad)


def test_wrong_aad_fails_closed(tmp_path):
    enc = _encryptor(tmp_path)
    payload = enc.encrypt("original text", aad=chunk_aad("doc-1", 0))
    with pytest.raises(InvalidTag):
        enc.decrypt(payload, aad=chunk_aad("doc-1", 1))  # wrong chunk index bound into AAD


def test_key_provider_persists_kek_across_instances(tmp_path):
    key_path = tmp_path / "master.key"
    enc1 = EnvelopeEncryptor(LocalKeyProvider(key_path))
    payload = enc1.encrypt("text", aad=b"aad")

    enc2 = EnvelopeEncryptor(LocalKeyProvider(key_path))  # reloads the same KEK from disk
    assert enc2.decrypt(payload, aad=b"aad") == "text"
