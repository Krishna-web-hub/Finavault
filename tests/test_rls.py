"""Tests for security/rls.py — tenant isolation enforced by PostgreSQL.

Split into two halves for a reason.

The pure tests run everywhere and cover the parts that are wrong in
*application* code: the org-id validation, the scoping context manager, and
the table/policy definitions.

The `postgres`-marked tests are the ones that actually matter, and they need
a real server — RLS is a database feature, and a policy that has never been
executed against Postgres is a policy nobody has verified. They skip
themselves when no server is reachable so a developer without Docker still
gets a green suite; CI runs them with a service container (see
.github/workflows/ci.yml). Every way this can be silently wrong — `ENABLE`
without `FORCE`, a policy dropped by a migration — looks identical to
working from inside the application, which is why they exist at all.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.exc import SQLAlchemyError

from finvault.db import documents_table, metadata
from finvault.security.rls import (
    ORG_SETTING,
    POLICY_NAME,
    RLS_TABLES,
    current_org,
    enable_row_level_security,
    install_org_scoping,
    org_scope,
    set_current_org,
    verify_isolation,
)

# The OWNER connection: creates tables and installs policies. Also a
# superuser in the local compose setup, which is precisely why it must not be
# the connection under test.
OWNER_DSN = "postgresql://finvault:finvault_dev_only@localhost:5433/finvault"

# The APPLICATION connection: a NOSUPERUSER, NOBYPASSRLS role that does not
# own the tables (see deploy/postgres/01-app-role.sql). Testing isolation
# through the owner connection would pass trivially and prove nothing — a
# superuser bypasses every policy, so the tests would be green on a database
# with no enforcement at all.
APP_DSN = "postgresql://finvault_app:finvault_app_dev_only@localhost:5433/finvault"


# --------------------------------------------------------------------------
# Pure tests — no database needed
# --------------------------------------------------------------------------


def test_every_table_under_policy_carries_an_org_column() -> None:
    """A policy keyed on a column the table does not have would fail at DDL
    time — but only on the deployment that enables RLS, which is production."""
    for table_name, column in RLS_TABLES.items():
        table = metadata.tables[table_name]
        assert column in table.c, f"{table_name} has no {column} column"


def test_the_covered_tables_are_exactly_the_org_scoped_ones() -> None:
    """Guards against a new tenant table being added to db.py and quietly
    left out of the policy set."""
    org_scoped = {name for name, table in metadata.tables.items() if "org_id" in table.c}
    assert set(RLS_TABLES) == org_scoped


def test_scoping_is_restored_after_the_block() -> None:
    with org_scope("org-a"):
        assert current_org() == "org-a"
    assert current_org() is None


def test_scoping_is_restored_even_when_the_block_raises() -> None:
    """Otherwise one failed request would leave its org bound for whatever
    ran next on the same worker."""
    with pytest.raises(ValueError), org_scope("org-a"):
        raise ValueError("boom")
    assert current_org() is None


def test_nested_scopes_restore_the_outer_org() -> None:
    with org_scope("org-a"):
        with org_scope("org-b"):
            assert current_org() == "org-b"
        assert current_org() == "org-a"


def test_scoping_to_none_means_see_nothing() -> None:
    """The fail-closed direction: a code path that forgets to declare its org
    reads back empty rather than reading everything."""
    with org_scope(None):
        assert current_org() is None


@pytest.mark.parametrize(
    "malformed",
    [
        "org'; DROP TABLE documents;--",
        "org\nid",
        "org id",
        "x" * 200,
        "",
    ],
)
def test_a_malformed_org_id_is_refused(malformed: str) -> None:
    """The value the entire isolation model rests on. It reaches Postgres
    through a bound parameter, so this is belt and braces — but the belt is
    cheap and the failure is total."""
    with pytest.raises(ValueError):
        set_current_org(malformed)


# --------------------------------------------------------------------------
# Database tests — the ones that actually verify the policy
# --------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Yields the *application* engine, with schema and policies installed by
    the owner engine first."""
    try:
        owner = create_engine(OWNER_DSN, future=True)
        with owner.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"no PostgreSQL available: {exc}")

    metadata.create_all(owner)
    enable_row_level_security(owner)

    try:
        app_engine = create_engine(APP_DSN, future=True)
        with app_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"finvault_app role not provisioned ({exc}); run deploy/postgres/01-app-role.sql")

    install_org_scoping(app_engine)
    yield app_engine

    # Leave the database usable for the next run: with policies still in
    # force and no org bound, a later unrelated test would read nothing and
    # fail for a reason that has nothing to do with what it tests.
    from finvault.security.rls import disable_row_level_security

    with owner.begin() as conn:
        conn.execute(text("DELETE FROM documents WHERE title LIKE 'rls-test-%'"))
    disable_row_level_security(owner)
    app_engine.dispose()
    owner.dispose()


def _insert_document(engine, org_id: str) -> str:
    document_id = str(uuid.uuid4())
    with org_scope(org_id), engine.begin() as conn:
        conn.execute(
            insert(documents_table).values(
                id=document_id,
                org_id=org_id,
                title=f"rls-test-{org_id}",
                classification="internal",
                created_at=text("now()"),
            )
        )
    return document_id


@pytest.mark.postgres
def test_policies_are_installed_and_forced(engine) -> None:
    """FORCE is the load-bearing half: plain ENABLE exempts the table owner,
    which is the account this application connects as. Without it the
    policies exist, appear in pg_policies, and apply to nobody."""
    results = verify_isolation(engine)
    assert results == dict.fromkeys(RLS_TABLES, True)

    with engine.connect() as conn:
        for table in RLS_TABLES:
            row = conn.execute(
                text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = :t"), {"t": table}
            ).first()
            assert row == (True, True), f"{table} is not both enabled and forced"


@pytest.mark.postgres
def test_a_query_without_an_org_returns_nothing(engine) -> None:
    """`current_setting(..., true)` is NULL when unset, and `org_id = NULL`
    is NULL — not TRUE — so the row is filtered out. This is the fail-closed
    behavior every non-request code path has to know about."""
    _insert_document(engine, "org-a")

    with org_scope(None), engine.connect() as conn:
        rows = conn.execute(select(documents_table)).all()
    assert rows == []


@pytest.mark.postgres
def test_one_org_cannot_read_anothers_rows(engine) -> None:
    """The whole point: the SELECT has no WHERE clause at all, and still
    cannot cross the tenant boundary."""
    _insert_document(engine, "org-a")
    _insert_document(engine, "org-b")

    with org_scope("org-a"), engine.connect() as conn:
        rows = conn.execute(select(documents_table)).all()
    assert rows and all(row.org_id == "org-a" for row in rows)


@pytest.mark.postgres
def test_a_forgotten_where_clause_leaks_nothing(engine) -> None:
    """This is the failure RLS exists to prevent, written as a test: raw SQL
    that a reviewer would flag, executed anyway, returning only the caller's
    own tenant's rows."""
    _insert_document(engine, "org-a")
    _insert_document(engine, "org-b")

    with org_scope("org-b"), engine.connect() as conn:
        rows = conn.execute(text("SELECT org_id FROM documents")).all()
    assert {row.org_id for row in rows} == {"org-b"}


@pytest.mark.postgres
def test_a_row_cannot_be_written_into_another_org(engine) -> None:
    """WITH CHECK covers writes. Without it a caller could INSERT a row
    stamped with another org's id — invisible to them afterwards, but
    present in that tenant's data."""
    with pytest.raises(SQLAlchemyError), org_scope("org-a"), engine.begin() as conn:
        conn.execute(
            insert(documents_table).values(
                id=str(uuid.uuid4()),
                org_id="org-b",
                title="rls-test-smuggled",
                classification="internal",
                created_at=text("now()"),
            )
        )


@pytest.mark.postgres
def test_the_org_setting_does_not_survive_into_the_next_transaction(engine) -> None:
    """SET LOCAL, not SET. The engine pools connections, so a session-level
    setting would ride the connection back into the pool and be inherited by
    whichever tenant checked it out next — the exact cross-tenant leak this
    module exists to prevent."""
    with org_scope("org-a"), engine.connect() as conn:
        assert conn.execute(text(f"SELECT current_setting('{ORG_SETTING}', true)")).scalar() == "org-a"

    with engine.connect() as conn:
        leaked = conn.execute(text(f"SELECT current_setting('{ORG_SETTING}', true)")).scalar()
    assert leaked in (None, ""), f"org setting leaked across transactions: {leaked!r}"


@pytest.mark.postgres
def test_a_superuser_connection_is_reported_as_unprotected() -> None:
    """The hazard that motivated `connection_role_bypasses_rls`: a superuser
    ignores row security entirely, so the policies are installed, pg_policies
    lists them, relforcerowsecurity is true — and nothing is filtered.

    verify_isolation must report that honestly rather than confirming the
    table flags and calling it protected, because the difference is invisible
    from inside the application.
    """
    from finvault.security.rls import connection_role_bypasses_rls

    try:
        owner = create_engine(OWNER_DSN, future=True)
        with owner.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"no PostgreSQL available: {exc}")

    bypasses, role = connection_role_bypasses_rls(owner)
    if not bypasses:
        pytest.skip(f"role {role} is not a superuser here, so there is no bypass to detect")

    enable_row_level_security(owner)
    assert verify_isolation(owner) == dict.fromkeys(RLS_TABLES, False)
    owner.dispose()


@pytest.mark.postgres
def test_the_application_role_does_not_bypass_policies(engine) -> None:
    """The other half: the role the app actually connects as must be subject
    to them."""
    from finvault.security.rls import connection_role_bypasses_rls

    bypasses, role = connection_role_bypasses_rls(engine)
    assert bypasses is False, f"application role {role} bypasses RLS"


@pytest.mark.postgres
def test_enabling_is_idempotent(engine) -> None:
    """Runs on every startup, so a second call must not fail on an existing
    policy — and must actually replace it, since Postgres has no
    CREATE OR REPLACE POLICY."""
    owner = create_engine(OWNER_DSN, future=True)
    enable_row_level_security(owner)
    enable_row_level_security(owner)
    # Verified through the application engine: the owner is a superuser here
    # and would report unprotected for that reason alone.
    assert verify_isolation(engine) == dict.fromkeys(RLS_TABLES, True)
    owner.dispose()

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM pg_policies WHERE tablename = 'documents' AND policyname = :p"),
            {"p": POLICY_NAME},
        ).scalar()
    assert count == 1
