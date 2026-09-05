"""Cloud KMS-backed KeyProvider.

`LocalKeyProvider` (security/encryption.py) keeps the KEK in a file on the
application host, and says in its own docstring that this is not a
substitute for a real KMS. This module is that substitute: the KEK is a
Cloud KMS CryptoKey that never leaves Google's HSM boundary. FinVault sends
a 32-byte DEK to be wrapped and receives it back unwrapped; the key material
that protects the corpus is never on disk, never in an environment
variable, and never in a backup of the application host.

What that buys beyond file-based custody:
  - KMS enforces IAM on every wrap/unwrap, so key use is authorized
    independently of whoever can read the filesystem.
  - Cloud Audit Logs record key use, outside FinVault's own audit chain —
    an attacker with database access cannot also forge the key-use record.
  - Rotation is a KMS operation. KMS decrypts with whichever key version
    wrapped a given DEK, so rotating does not require re-wrapping the
    existing corpus.

**Call-volume note.** `EnvelopeEncryptor` wraps one DEK per chunk, so a
direct KMS provider costs one round trip per chunk: a 200-chunk document is
200 wraps at ingest, and each query unwraps one DEK per retrieved chunk.
GCP KMS has neither a batch encrypt nor AWS's GenerateDataKey (which returns
plaintext and wrapped DEK in a single call), so that count is inherent to
the design rather than something to optimize away. Ingestion absorbs it
(already asynchronous, already dominated by embedding). Retrieval does not,
which is what `_DekCache` below is for.
"""

from __future__ import annotations

import threading
import time

from finvault.config import settings
from finvault.errors import ConfigurationError
from finvault.observability import extra_fields, get_logger
from finvault.security.encryption import KeyProvider, LocalKeyProvider

logger = get_logger(__name__)


class _DekCache:
    """Bounded, TTL'd cache of unwrapped DEKs, keyed by the wrapped bytes.

    Retrieval re-reads the same chunks constantly — a popular document's
    DEK would otherwise be unwrapped through KMS on every query that
    touches it. Caching collapses steady-state reads to roughly zero KMS
    calls.

    Only unwraps are cached. Every wrap produces a fresh random DEK, so
    there is nothing to reuse on that side.

    The security cost is explicit and bounded: while an entry is live, a
    decrypt happens without KMS authorizing it, so revoking FinVault's IAM
    access takes effect only after the TTL expires. That is the reason for
    a short default TTL and a hard size cap rather than an unbounded map.
    Plaintext DEKs in process memory are not a new exposure — decrypt()
    already holds them transiently — but a cache holds them longer, so the
    cap matters.
    """

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        # Insertion-ordered, so the oldest entry is the first key — enough
        # for FIFO eviction without pulling in an LRU dependency.
        self._entries: dict[bytes, tuple[bytes, float]] = {}
        # Retrieval runs in FastAPI's threadpool and ingestion via
        # to_thread, so this is touched concurrently.
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0 and self._ttl_seconds > 0

    def get(self, wrapped: bytes) -> bytes | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(wrapped)
            if entry is None:
                return None
            dek, expires_at = entry
            if expires_at <= now:
                # Expired entries are dropped on read rather than swept: the
                # cache is small and bounded, so a background sweeper would
                # be machinery without a purpose.
                del self._entries[wrapped]
                return None
            return dek

    def put(self, wrapped: bytes, dek: bytes) -> None:
        if not self.enabled:
            return
        with self._lock:
            if wrapped in self._entries:
                del self._entries[wrapped]
            elif len(self._entries) >= self._max_entries:
                oldest = next(iter(self._entries))
                del self._entries[oldest]
            self._entries[wrapped] = (dek, time.monotonic() + self._ttl_seconds)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class GcpKmsKeyProvider(KeyProvider):
    """Wraps and unwraps DEKs with a Google Cloud KMS symmetric CryptoKey.

    `key_name` is the full resource path:
        projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>

    Note it names the *CryptoKey*, not a version. KMS encrypts with the
    key's primary version and decrypts with whichever version produced a
    given ciphertext, which is what makes rotation a KMS-side operation
    with no re-wrapping of the existing corpus.

    `client` is injectable so tests exercise this class without GCP
    credentials or network — the same seam every other external dependency
    in this codebase uses.
    """

    def __init__(
        self,
        key_name: str,
        *,
        client: object | None = None,
        cache_max_entries: int | None = None,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        if not key_name:
            raise ValueError("GcpKmsKeyProvider requires a KMS key resource name")
        self._key_name = key_name
        self._client = client if client is not None else _build_kms_client()
        self._cache = _DekCache(
            max_entries=(
                cache_max_entries if cache_max_entries is not None else settings.finvault_kms_dek_cache_max_entries
            ),
            ttl_seconds=(
                cache_ttl_seconds if cache_ttl_seconds is not None else settings.finvault_kms_dek_cache_ttl_seconds
            ),
        )

    def wrap_key(self, data_key: bytes) -> bytes:
        request: dict[str, object] = {"name": self._key_name, "plaintext": data_key}
        crc = _crc32c(data_key)
        if crc is not None:
            # KMS verifies this server-side and rejects a payload corrupted
            # in transit, rather than silently wrapping the wrong bytes —
            # which would surface much later as an undecryptable chunk.
            request["plaintext_crc32c"] = crc

        response = self._client.encrypt(request=request)

        if crc is not None and not getattr(response, "verified_plaintext_crc32c", True):
            raise KmsIntegrityError("KMS did not verify the plaintext checksum for a wrap request")
        return response.ciphertext

    def unwrap_key(self, wrapped_key: bytes) -> bytes:
        cached = self._cache.get(wrapped_key)
        if cached is not None:
            return cached

        request: dict[str, object] = {"name": self._key_name, "ciphertext": wrapped_key}
        crc = _crc32c(wrapped_key)
        if crc is not None:
            request["ciphertext_crc32c"] = crc

        response = self._client.decrypt(request=request)

        plaintext = response.plaintext
        expected = getattr(response, "plaintext_crc32c", None)
        actual = _crc32c(plaintext)
        if expected is not None and actual is not None and int(expected) != int(actual):
            # Fail loudly: a corrupted DEK would otherwise produce a GCM
            # authentication failure much further downstream, where it reads
            # as data corruption rather than a transport problem.
            raise KmsIntegrityError("KMS response checksum mismatch when unwrapping a data key")

        self._cache.put(wrapped_key, plaintext)
        return plaintext

    def invalidate_cache(self) -> None:
        """Drops every cached DEK, forcing the next decrypt of each chunk
        back through KMS. Exists so revoking KMS access can be made to take
        effect immediately rather than after the TTL.
        """
        self._cache.clear()


class KmsIntegrityError(Exception):
    """A KMS response failed its checksum verification."""


def _crc32c(data: bytes) -> int | None:
    """CRC32C of `data`, or None when google-crc32c isn't installed.

    Checksums are optional in the KMS API, so their absence degrades to
    "no transit-corruption detection" rather than breaking encryption —
    the library ships with the `gcpkms` extra, so a correct install has it.
    """
    try:
        import google_crc32c
    except ImportError:
        return None
    return int(google_crc32c.Checksum(data).hexdigest(), 16)


def _build_kms_client() -> object:
    try:
        from google.cloud import kms
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "GcpKmsKeyProvider requires the google-cloud-kms package. Install it with: pip install 'finvault[gcpkms]'"
        ) from exc
    return kms.KeyManagementServiceClient()


def build_key_provider() -> KeyProvider:
    """The KeyProvider this deployment should use, chosen from configuration.

    Unlike `build_cache`, this deliberately does **not** fall back to the
    local provider when the configured backend fails. A cache that is
    unreachable costs performance; a KEK that silently changes identity
    costs the corpus — every chunk wrapped under the old KEK becomes
    undecryptable, and every chunk written under the new one is protected
    by weaker custody than the operator asked for. Failing startup is the
    only safe outcome.
    """
    backend = settings.finvault_key_provider.strip().lower()

    if backend == "local":
        logger.info("key_provider_selected", extra=extra_fields(backend="local"))
        return LocalKeyProvider(settings.finvault_master_key_path)

    if backend == "gcp_kms":
        key_name = settings.finvault_gcp_kms_key_name
        if not key_name:
            raise ConfigurationError(
                "FINVAULT_KEY_PROVIDER=gcp_kms requires FINVAULT_GCP_KMS_KEY_NAME "
                "(projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>)",
                context={"key_provider": backend},
            )
        logger.info("key_provider_selected", extra=extra_fields(backend="gcp_kms", key_name=key_name))
        return GcpKmsKeyProvider(key_name)

    raise ConfigurationError(
        f"Unknown FINVAULT_KEY_PROVIDER {backend!r}. Supported values: local, gcp_kms",
        context={"key_provider": backend},
    )
