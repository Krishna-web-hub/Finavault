"""PostgreSQL Row Level Security — tenant isolation the database enforces.

Everything else in this system filters by `org_id` in application code:
every query adds a `WHERE org_id = ...`, and every one of them is correct
today. The problem is that correctness is re-established by hand at each
call site, so the isolation holds only for as long as nobody writes a query
that forgets. RLS moves the guarantee under the application: with a policy
in place, a query that omits the org filter returns *no rows* rather than
another tenant's, and there is no way to write the leaking query at all.

That is the whole point of doing this at the database. Application filters
are a convention; a policy is a mechanism.

## How it works here

1. `enable_row_level_security()` installs, per table, a policy of the form
   `org_id = current_setting('finvault.org_id', true)`.
2. `install_org_scoping()` registers a SQLAlchemy `begin` listener that
   issues `SET LOCAL finvault.org_id` from a context variable at the start
   of every transaction.
3. `api/auth.py` puts the verified token's org into that context variable,
   so the setting is a value the server derived from a signature — never
   something a caller supplied.

Two consequences worth understanding before enabling this:

**No org set means no rows, not all rows.** `current_setting(..., true)`
returns NULL when unset, and `org_id = NULL` is NULL, which is not TRUE, so
the row is filtered out. Any code path that reads these tables outside a
request must declare its org explicitly with `org_scope()` — background
jobs, scripts, and the demo in `scripts/` all do. This is the fail-closed
direction: a forgotten scope shows nothing, rather than showing everything.

**`FORCE ROW LEVEL SECURITY` is not optional**, and neither is the role the
application connects as. There are two independent ways to install these
policies and enforce nothing, and both look identical to a working system:

1. Plain `ENABLE` exempts the table's *owner*, and an application that
   created its own tables owns them. `FORCE` closes that.
2. A **superuser, or any role with `BYPASSRLS`, ignores row security
   entirely** — no `FORCE` and no policy applies to it. This is the default
   state of a database created the obvious way: `POSTGRES_USER` in the
   official Postgres image creates a superuser, so an application reusing
   that account has RLS on and doing nothing.

Connect as a dedicated non-superuser role that does not own the tables (see
`deploy/postgres/01-app-role.sql`, and `postgres.user` in the Helm values).
`verify_isolation()` checks both conditions, and `api/main.py` refuses to
start when either fails — because from inside the application, unenforced
policies are indistinguishable from enforced ones.

## What is covered, and what is not

Covered: `documents`, `review_queue`, `graph_nodes`, `graph_edges` — every
table carrying an `org_id`.

Not covered, deliberately:

- `audit_log` has no `org_id` and is one hash chain across the deployment.
  A policy over it would hide entries from `verify_chain()`, which must read
  every row in sequence or it cannot detect tampering — the integrity
  property would be traded for an isolation property the table does not
  need, since no route reads it.
- `sessions` is scoped by `user_id`, a server-generated UUID, and holds one
  user's own conversation turns rather than tenant data. Bringing it under
  RLS needs an `org_id` column and a signature change through
  `SessionStore`; it is a genuine follow-up, not something this module
  quietly claims to have done.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from finvault.observability import extra_fields, get_logger

logger = get_logger(__name__)

# The Postgres run-time parameter the policies read. A custom, dotted name
# is required — Postgres only permits `SET` on names containing a dot for
# parameters it does not itself define.
ORG_SETTING = "finvault.org_id"

# Tables under RLS, and the column each policy keys on. Adding a table means
# adding it here; the DDL below is generated from this, so a new tenant table
# cannot be half-covered.
RLS_TABLES: dict[str, str] = {
    "documents": "org_id",
    "review_queue": "org_id",
    "graph_nodes": "org_id",
    "graph_edges": "org_id",
}

POLICY_NAME = "finvault_tenant_isolation"

# An org id reaches Postgres through a bound parameter, never string
# interpolation — but it also ends up in a session setting that policies
# compare against, so it is validated as well. Belt and braces on the one
# value the entire isolation model rests on.
_ORG_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

_current_org: contextvars.ContextVar[str | None] = contextvars.ContextVar("finvault_rls_org", default=None)


def current_org() -> str | None:
    return _current_org.get()


def set_current_org(org_id: str | None) -> None:
    """Sets the org for subsequent transactions on this context.

    Called from `api/auth.py` once a token verifies. Like the logging
    context it sits beside, this must be set from async code running in the
    request's own task — a value set inside FastAPI's threadpool would be
    discarded (see `observability.add_request_context`).
    """
    if org_id is not None and not _ORG_ID_RE.match(org_id):
        raise ValueError(f"Refusing to scope a database session to a malformed org id: {org_id!r}")
    _current_org.set(org_id)


@contextmanager
def org_scope(org_id: str | None) -> Iterator[None]:
    """Runs a block with an explicit org, restoring the previous one after.

    For every path that has no request behind it: ingestion scripts, the
    demo, evaluation harnesses, and background work. Passing `None` is the
    way to say "this block should see nothing", which is what a policy-
    covered table returns without a scope.
    """
    if org_id is not None and not _ORG_ID_RE.match(org_id):
        raise ValueError(f"Refusing to scope a database session to a malformed org id: {org_id!r}")
    token = _current_org.set(org_id)
    try:
        yield
    finally:
        _current_org.reset(token)


def install_org_scoping(engine: Engine) -> None:
    """Makes every transaction on `engine` carry the current org.

    A `begin` listener rather than a connection-checkout one, and
    `SET LOCAL` rather than `SET`, because the engine pools connections: a
    session-level setting would outlive the transaction, ride the connection
    back into the pool, and be inherited by whichever tenant's request
    checked it out next. That is the exact cross-tenant leak this module
    exists to prevent, so the setting is bound to the transaction and
    discarded when it ends.
    """
    if getattr(engine, "_finvault_org_scoping", False):
        return

    @event.listens_for(engine, "begin")
    def _apply_org_scope(conn) -> None:  # type: ignore[no-untyped-def]
        # set_config(..., is_local => true) is SET LOCAL with a bindable
        # value; the plain SET statement takes only a literal, which would
        # mean interpolating an identity value into SQL.
        conn.exec_driver_sql(
            "SELECT set_config(%s, %s, true)",
            (ORG_SETTING, current_org() or ""),
        )

    engine._finvault_org_scoping = True  # type: ignore[attr-defined]
    logger.info("rls_org_scoping_installed", extra=extra_fields(setting=ORG_SETTING))


def enable_row_level_security(engine: Engine) -> None:
    """Installs the policies. Idempotent — safe on every startup.

    `DROP POLICY IF EXISTS` before `CREATE POLICY` so a changed policy
    definition actually takes effect on redeploy; Postgres has no
    `CREATE OR REPLACE POLICY`, and without the drop an old policy would
    silently persist while the new definition looked deployed.
    """
    with engine.begin() as conn:
        for table, column in RLS_TABLES.items():
            conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
            # Without FORCE, the table owner — which is the account this
            # application connects as — bypasses every policy below.
            conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
            conn.execute(text(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}"))
            conn.execute(
                text(
                    f"CREATE POLICY {POLICY_NAME} ON {table} "
                    f"USING ({column} = current_setting('{ORG_SETTING}', true)) "
                    # WITH CHECK covers writes: without it a caller could
                    # INSERT a row stamped with another org's id — invisible
                    # to them afterwards, but present in that tenant's data.
                    f"WITH CHECK ({column} = current_setting('{ORG_SETTING}', true))"
                )
            )
    logger.info("rls_enabled", extra=extra_fields(tables=sorted(RLS_TABLES), policy=POLICY_NAME))


def disable_row_level_security(engine: Engine) -> None:
    """Removes the policies. For local development and for backing the
    feature out without a database rebuild — never called automatically.
    """
    with engine.begin() as conn:
        for table in RLS_TABLES:
            conn.execute(text(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}"))
            conn.execute(text(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"))
            conn.execute(text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))
    logger.warning("rls_disabled", extra=extra_fields(tables=sorted(RLS_TABLES)))


def connection_role_bypasses_rls(engine: Engine) -> tuple[bool, str]:
    """Whether the role this engine connects as is exempt from every policy.

    **This is the check that matters most, and the easiest one to skip.** A
    superuser, or any role with `BYPASSRLS`, ignores row security entirely —
    policies are installed, `pg_policies` lists them, `relforcerowsecurity`
    is true, and not one row is ever filtered. Nothing about the application
    looks different.

    It is also the default state of a database created the obvious way:
    `POSTGRES_USER` in the official Postgres image creates a superuser, so a
    deployment that reuses that account for the application has RLS switched
    on and doing nothing. Returns (bypasses, role_name).
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        ).first()
    if row is None:
        return False, "unknown"
    role, is_super, can_bypass = row
    return bool(is_super or can_bypass), str(role)


def verify_isolation(engine: Engine) -> dict[str, bool]:
    """Confirms tenant isolation is actually in effect. Returns table -> protected.

    Worth calling at startup because every way this can be wrong is silent:

    - `ENABLE` without `FORCE` leaves the table's owner exempt, and the
      application usually *is* the owner;
    - a policy dropped by a migration leaves the table open while the code
      that expects protection carries on unchanged;
    - and the connecting role may bypass row security altogether, which
      makes the other two checks pass while nothing is enforced.

    A bypassing role reports **every** table as unprotected, because that is
    the truth: the policies exist and apply to nobody.
    """
    bypasses, role = connection_role_bypasses_rls(engine)
    if bypasses:
        logger.error(
            "rls_ineffective_connection_role",
            extra=extra_fields(
                role=role,
                reason="role is a superuser or has BYPASSRLS; row security policies do not apply to it",
                remedy="connect as a dedicated non-superuser application role that does not own these tables",
            ),
        )
        return dict.fromkeys(RLS_TABLES, False)

    results: dict[str, bool] = {}
    with engine.connect() as conn:
        for table in RLS_TABLES:
            row = conn.execute(
                text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = :t"),
                {"t": table},
            ).first()
            has_policy = conn.execute(
                text("SELECT 1 FROM pg_policies WHERE tablename = :t AND policyname = :p"),
                {"t": table, "p": POLICY_NAME},
            ).first()
            results[table] = bool(row and row[0] and row[1] and has_policy)

    unprotected = sorted(t for t, ok in results.items() if not ok)
    if unprotected:
        logger.error("rls_verification_failed", extra=extra_fields(unprotected_tables=unprotected, role=role))
    else:
        logger.info("rls_verified", extra=extra_fields(tables=sorted(results), role=role))
    return results


__all__ = [
    "ORG_SETTING",
    "RLS_TABLES",
    "POLICY_NAME",
    "current_org",
    "set_current_org",
    "org_scope",
    "install_org_scoping",
    "enable_row_level_security",
    "disable_row_level_security",
    "verify_isolation",
    "connection_role_bypasses_rls",
]
