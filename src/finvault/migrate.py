"""Schema migration entry point — `finvault-migrate` on the PATH.

This is what the Helm pre-install Job runs, and it is the only supported way
a non-development database gets its schema. It exists as a module rather
than an `alembic upgrade head` shell invocation for two reasons:

- **The revisions ship inside the package.** `script_location` is resolved
  from `finvault.__file__`, so this works from the application image, from
  an installed wheel, and from a checked-out repo without any of them
  needing the same working directory or an `alembic.ini` on disk.
- **Policy installation belongs in the same step.** Creating the tables and
  installing the Row Level Security policies must happen together and under
  the owning role; splitting them across a migration and a startup path is
  how a deployment ends up with tables that exist and policies that do not.

Run against a database owned by the migration role, not the role the
application connects as — see `security/rls.py` for why those differ.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from finvault.config import settings
from finvault.observability import configure_logging, extra_fields, get_logger
from finvault.security.rls import enable_row_level_security, verify_isolation

logger = get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    """An Alembic Config that does not depend on the current directory."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url or settings.postgres_dsn)
    return config


def upgrade_to_head(url: str | None = None) -> None:
    command.upgrade(alembic_config(url), "head")


def current_revision(url: str | None = None) -> str | None:
    """The revision a database is stamped at, or None if it has never been
    migrated. Used by tests and by anyone diagnosing a version mismatch."""
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    engine = create_engine(url or settings.postgres_dsn, future=True)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def main() -> None:
    configure_logging()
    logger.info("migration_begin", extra=extra_fields(revision_before=current_revision()))

    upgrade_to_head()
    logger.info("migration_complete", extra=extra_fields(revision=current_revision()))

    if settings.finvault_enable_rls:
        from sqlalchemy import create_engine

        engine = create_engine(settings.postgres_dsn, future=True)
        try:
            enable_row_level_security(engine)
            # Only the table flags are meaningful from here: this process
            # connects as the owner, which bypasses policies by design, so
            # the authoritative end-to-end check is the application's own
            # startup verification under its real role (see api/main.py).
            logger.info("rls_installed", extra=extra_fields(tables=sorted(verify_isolation(engine))))
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
