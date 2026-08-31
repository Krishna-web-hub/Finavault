"""Bounded, always-cleaned-up handling of uploaded files.

This existed twice, inline, in `routes.py` — once in `POST /documents` and
once in `POST /documents/classification-suggestion` — with two size checks,
two temp-file lifecycles, and two chances for the cleanup to drift out of
sync with the failure paths. One of the two had already grown a subtly
different error message for the same condition.

The size cap is enforced twice on purpose:

1. against `UploadFile.size`, the client's declared Content-Length, which
   is free to check but trivially wrong or absent; and
2. against the bytes actually written, which is the one that counts — a
   client can declare 1 KB and stream 10 GB.

The second check runs per chunk while streaming, so an oversized upload is
rejected as soon as it crosses the line, not after the whole thing is on
disk.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import UploadFile

from finvault.config import settings
from finvault.errors import PayloadTooLargeError
from finvault.observability import extra_fields, get_logger

logger = get_logger(__name__)

_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def spooled_upload(file: UploadFile, *, max_bytes: int | None = None) -> AsyncIterator[Path]:
    """Streams an upload to a temp file and yields its path, deleting it on
    the way out — on success, on error, and on `PayloadTooLargeError`.

        async with spooled_upload(file) as path:
            document = await asyncio.to_thread(pipeline.ingest_file, path, ...)

    Raises `PayloadTooLargeError` (413) if the upload exceeds `max_bytes`,
    defaulting to `settings.finvault_max_upload_size_mb`. The partial file
    is removed before the exception propagates, so a client hammering an
    oversized upload cannot fill the disk.
    """
    limit = max_bytes if max_bytes is not None else settings.finvault_max_upload_size_mb * 1024 * 1024

    if file.size is not None and file.size > limit:
        # Declared size — cheap to check and rejects the common case before
        # a single byte is read. Not trusted: the streaming check below is
        # the enforcing one.
        raise _too_large(declared_bytes=file.size, limit=limit)

    suffix = Path(file.filename or "").suffix
    tmp_path: Path | None = None
    try:
        written = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise _too_large(written_bytes=written, limit=limit)
                tmp.write(chunk)
        logger.debug("upload_spooled", extra=extra_fields(bytes=written, suffix=suffix))
        yield tmp_path
    finally:
        # One cleanup for every exit path, which is the entire reason this
        # is a context manager rather than a helper function.
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _too_large(
    *, limit: int, declared_bytes: int | None = None, written_bytes: int | None = None
) -> PayloadTooLargeError:
    observed = declared_bytes if declared_bytes is not None else (written_bytes or 0)
    limit_mb = limit / (1024 * 1024)
    return PayloadTooLargeError(
        f"Upload of {observed} bytes exceeds the {limit} byte limit",
        context={
            "observed_bytes": observed,
            "limit_bytes": limit,
            "detected_by": "declared_size" if declared_bytes is not None else "streamed_bytes",
        },
        user_message=(
            f"File size ({observed / (1024 * 1024):.1f} MB) exceeds the maximum upload limit of {limit_mb:.0f} MB."
        ),
    )


__all__ = ["spooled_upload"]
