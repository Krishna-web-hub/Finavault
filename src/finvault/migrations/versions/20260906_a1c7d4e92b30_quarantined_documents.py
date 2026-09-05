"""quarantined documents

Adds the table behind `security/quarantine.py` — the response half of
prompt-injection handling. Detection already existed
(`detect_injection_attempt`); until now nothing could act on it, so a
poisoned document stayed retrievable forever.

Row Level Security is deliberately absent here, exactly as in the baseline
migration: policies are re-derived idempotently from `RLS_TABLES` in
`security/rls.py` on every deploy, and `quarantined_documents` is
registered there. `finvault-migrate` runs both steps in order.

Revision ID: a1c7d4e92b30
Revises: 4438101f076d
Create Date: 2026-09-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c7d4e92b30"
down_revision: str | Sequence[str] | None = "4438101f076d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "quarantined_documents",
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("org_id", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("quarantined_by", sa.String(), nullable=False),
        sa.Column("quarantined_at", sa.Float(), nullable=False),
        sa.Column("released_by", sa.String(), nullable=True),
        sa.Column("released_at", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index(op.f("ix_quarantined_documents_org_id"), "quarantined_documents", ["org_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_quarantined_documents_org_id"), table_name="quarantined_documents")
    op.drop_table("quarantined_documents")
