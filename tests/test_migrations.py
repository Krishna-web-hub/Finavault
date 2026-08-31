"""Tests for the Alembic migration path.

The claim these defend is narrow and specific: `finvault-migrate` builds the
same schema that `finvault.db.metadata` describes. That equivalence is the
whole reason the application can keep defining its tables in `db.py` while
production gets them from reviewed revisions — if the two ever diverge, the
divergence shows up as a column that exists on a developer's laptop and
nowhere else, which is the failure mode migrations were added to end.

The comparison test needs a real PostgreSQL: it applies both paths to two
scratch databases and diffs the reflected result, and there is no way to do
that against SQLite or a mock without the dialect differences swamping the
answer. It is marked `postgres` and skips cleanly without a server; CI runs
it with a service container. The graph tests beside it need no server.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

from finvault.config import settings
from finvault.db import init_db, metadata
from finvault.migrate import alembic_config, current_revision, upgrade_to_head


def _script_directory():
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config(url="postgresql://unused/unused"))


# --- revision graph (no server needed) ---


def test_revisions_are_packaged_with_the_application() -> None:
    """The Helm Job runs `finvault-migrate` from the application image, which
    holds the installed package and not the repo layout. If the revisions
    stop resolving relative to `finvault/`, that Job fails at deploy time
    with an empty version history — which Alembic reports as "already up to
    date" rather than as an error.
    """
    # walk_revisions(), not get_revisions("base") — "base" is Alembic's
    # null revision and resolves to an empty tuple even when the directory
    # is fully populated.
    revisions = list(_script_directory().walk_revisions())
    assert revisions, "no revisions found next to the finvault package"


def test_revision_history_has_exactly_one_head() -> None:
    """Two heads mean two branches of schema history and an `upgrade head`
    that refuses to run. It happens when two branches each add a revision
    against the same parent, and the merge is easy — but only if someone
    notices before the deploy does.
    """
    heads = _script_directory().get_heads()
    assert len(heads) == 1, f"expected a single migration head, found {heads}"


# --- schema equivalence (needs Postgres) ---


@pytest.fixture(scope="module")
def admin_engine():
    engine = create_engine(settings.postgres_dsn, isolation_level="AUTOCOMMIT", future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError:
        pytest.skip(f"Postgres not reachable at {settings.postgres_dsn!r} — skipping migration schema tests")
    return engine


@pytest.fixture
def scratch_databases(admin_engine):
    """Two throwaway databases, dropped afterwards.

    Real databases rather than schemas within one: `upgrade head` and
    `create_all` both operate on a whole database, and running them into
    namespaces of a shared one would not exercise the same code path.
    """
    suffix = uuid.uuid4().hex[:12]
    names = (f"fv_mig_{suffix}", f"fv_meta_{suffix}")
    with admin_engine.connect() as conn:
        for name in names:
            conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        base = settings.postgres_dsn.rsplit("/", 1)[0]
        yield tuple(f"{base}/{name}" for name in names)
    finally:
        with admin_engine.connect() as conn:
            for name in names:
                conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _describe(url: str) -> dict:
    """Tables, columns and indexes as the server actually reports them."""
    engine = create_engine(url, future=True)
    try:
        inspector = inspect(engine)
        return {
            table: {
                "columns": {c["name"]: (str(c["type"]), bool(c["nullable"])) for c in inspector.get_columns(table)},
                "indexes": sorted(
                    (i["name"], tuple(i["column_names"]), bool(i["unique"])) for i in inspector.get_indexes(table)
                ),
                "primary_key": tuple(inspector.get_pk_constraint(table)["constrained_columns"]),
            }
            # alembic_version exists only on the migrated side by
            # construction — it is Alembic's bookkeeping, not application
            # schema, and comparing it would fail the test for the one
            # difference that is supposed to be there.
            for table in sorted(set(inspector.get_table_names()) - {"alembic_version"})
        }
    finally:
        engine.dispose()


@pytest.mark.postgres
def test_migrations_produce_the_same_schema_as_the_models(scratch_databases) -> None:
    migrated_url, metadata_url = scratch_databases

    upgrade_to_head(migrated_url)

    metadata_engine = create_engine(metadata_url, future=True)
    try:
        init_db(metadata_engine)
    finally:
        metadata_engine.dispose()

    assert _describe(migrated_url) == _describe(metadata_url)


@pytest.mark.postgres
def test_every_table_the_application_queries_exists_after_migrating(scratch_databases) -> None:
    """A narrower assertion than the diff above, kept because it fails with a
    readable message. The diff says "these two dicts differ"; this says which
    table the application would have gone looking for and not found.
    """
    migrated_url, _ = scratch_databases
    upgrade_to_head(migrated_url)

    engine = create_engine(migrated_url, future=True)
    try:
        present = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert set(metadata.tables) <= present


@pytest.mark.postgres
def test_a_migrated_database_reports_its_revision(scratch_databases) -> None:
    """An unmigrated database reports None, which is how `finvault-migrate`
    logs "this was a fresh database" versus "this was an upgrade" — and how
    an operator tells a database that is behind from one that was never
    stamped at all.
    """
    migrated_url, untouched_url = scratch_databases

    assert current_revision(untouched_url) is None

    upgrade_to_head(migrated_url)
    assert current_revision(migrated_url) == _script_directory().get_current_head()
