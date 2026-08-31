"""Alembic environment.

Two things here differ from the generated template, both deliberate:

- **The URL comes from `finvault.config.settings`, not `alembic.ini`.** The
  DSN is a deployment secret that already reaches the app through the
  environment; duplicating it into a checked-in ini file would be a second
  place to get it wrong, and the wrong one would be the one in git.
- **`target_metadata` is `finvault.db.metadata`** — the same object the
  application builds its queries from. That is what makes `--autogenerate`
  meaningful and what the CI drift check compares against (see
  .github/workflows/ci.yml): if a table is edited in db.py and no migration
  is written, autogenerate produces a non-empty diff and the job fails.

Note that migrations run as the *owning* role and therefore bypass Row Level
Security. That is required — installing a policy needs ownership — and is
also why the application connects as a different, non-owner role. See
`security/rls.py`.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from finvault.config import settings
from finvault.db import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def _url() -> str:
    """`-x url=...` wins, then the ini, then the app's own settings.

    The override exists for the one case the settings cannot serve: running a
    migration against a database that is not the one this process is
    configured to talk to — the CI drift check's scratch database, or a
    one-off against a restored snapshot.
    """
    return (
        context.get_x_argument(as_dictionary=True).get("url")
        or config.get_main_option("sqlalchemy.url", default=None)
        or settings.postgres_dsn
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    Kept because a reviewed, human-read `alembic upgrade head --sql` is how
    a change reaches a production database in environments where the
    application's role is not allowed to run DDL at all.
    """
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool, future=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without these two, autogenerate compares table and column
            # *names* only — a String widened to Text, or a default added in
            # db.py, would produce an empty diff and the CI drift check would
            # pass on a schema that no longer matches the models.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
