"""Tests for api/uploads.py — the bounded, self-cleaning upload helper.

Two routes stream user-supplied files to disk. What matters is that the
size cap cannot be talked past and that the temp file is gone afterwards on
every path, including the ones that raise — the failure modes that a
duplicated inline implementation kept getting subtly wrong.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import UploadFile

from finvault.api.uploads import spooled_upload
from finvault.errors import PayloadTooLargeError


def _upload(content: bytes, *, filename: str = "report.txt", declared_size: int | None = None) -> UploadFile:
    upload = UploadFile(filename=filename, file=io.BytesIO(content))
    if declared_size is not None:
        # UploadFile.size mirrors the Content-Length a client sent, which is
        # a claim, not a measurement — overridable here for exactly that
        # reason.
        upload.size = declared_size
    return upload


def _run(coro):
    """spooled_upload is an async context manager and this project has no
    pytest-asyncio dependency, so each scenario is a coroutine driven by
    asyncio.run — one event loop per test, nothing shared between them."""
    import asyncio

    return asyncio.run(coro)


def test_content_is_written_and_the_temp_file_is_removed_afterwards() -> None:
    seen: dict = {}

    async def scenario() -> None:
        async with spooled_upload(_upload(b"Q3 revenue was $10M.")) as path:
            seen["path"] = path
            seen["content"] = path.read_text()
            seen["suffix"] = path.suffix

    _run(scenario())
    assert seen["content"] == "Q3 revenue was $10M."
    # The suffix is preserved so loaders can dispatch on file type.
    assert seen["suffix"] == ".txt"
    assert not seen["path"].exists()


def test_a_declared_size_over_the_limit_is_rejected_before_any_read() -> None:
    async def scenario() -> None:
        async with spooled_upload(_upload(b"x" * 10, declared_size=10_000_000), max_bytes=100):
            pytest.fail("body should never have been entered")

    with pytest.raises(PayloadTooLargeError) as exc_info:
        _run(scenario())
    assert exc_info.value.context["detected_by"] == "declared_size"
    assert exc_info.value.http_status == 413


def test_a_lying_declared_size_is_caught_while_streaming() -> None:
    """The check that actually enforces the limit: a client can declare 1 KB
    and stream far more."""

    async def scenario() -> None:
        async with spooled_upload(_upload(b"x" * 5000, declared_size=10), max_bytes=100):
            pytest.fail("body should never have been entered")

    with pytest.raises(PayloadTooLargeError) as exc_info:
        _run(scenario())
    assert exc_info.value.context["detected_by"] == "streamed_bytes"


def test_the_partial_file_is_deleted_when_an_upload_is_rejected(tmp_path, monkeypatch) -> None:
    """Otherwise a client hammering an oversized upload fills the disk."""
    created: list[Path] = []
    import tempfile as tempfile_module

    real_named_temp = tempfile_module.NamedTemporaryFile

    def tracking_named_temp(*args, **kwargs):
        handle = real_named_temp(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr("finvault.api.uploads.tempfile.NamedTemporaryFile", tracking_named_temp)

    async def scenario() -> None:
        async with spooled_upload(_upload(b"x" * 5000), max_bytes=100):
            pytest.fail("body should never have been entered")

    with pytest.raises(PayloadTooLargeError):
        _run(scenario())

    assert created, "expected a temp file to have been created"
    assert not created[0].exists()


def test_the_temp_file_is_removed_when_the_body_itself_raises() -> None:
    """Cleanup is in a `finally`, so an ingestion failure inside the block
    does not leave the upload behind."""
    seen: dict = {}

    async def scenario() -> None:
        async with spooled_upload(_upload(b"data")) as path:
            seen["path"] = path
            raise RuntimeError("ingestion blew up")

    with pytest.raises(RuntimeError):
        _run(scenario())
    assert not seen["path"].exists()


def test_the_client_facing_message_states_sizes_but_no_internals() -> None:
    async def scenario() -> None:
        async with spooled_upload(_upload(b"x" * 5000), max_bytes=1024):
            pass

    with pytest.raises(PayloadTooLargeError) as exc_info:
        _run(scenario())
    message = exc_info.value.user_message
    assert "MB" in message
    assert "/tmp" not in message
