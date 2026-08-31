"""Role-based clearance and organization isolation checks.

Classification and role rank are the real access-control primitive in
FinVault (see models.py) — every retrieval path calls these before any
plaintext is produced.
"""

from __future__ import annotations

from finvault.errors import AccessDeniedError
from finvault.models import CLASSIFICATION_MIN_ROLE, ROLE_RANK, Classification, Role

# Re-exported so `from finvault.security.access_control import AccessDeniedError`
# keeps working; the class itself now lives in finvault/errors.py with every
# other failure mode. Same for the other security modules.
__all__ = ["AccessDeniedError", "check_clearance", "require_clearance", "require_same_org"]


def check_clearance(role: Role, classification: Classification) -> bool:
    return ROLE_RANK[role] >= ROLE_RANK[CLASSIFICATION_MIN_ROLE[classification]]


def require_clearance(role: Role, classification: Classification, *, resource: str = "resource") -> None:
    if not check_clearance(role, classification):
        required = CLASSIFICATION_MIN_ROLE[classification]
        raise AccessDeniedError(
            f"Role '{role.value}' lacks clearance for {classification.value}-classified "
            f"{resource} (requires at least '{required.value}')",
            context={
                "role": role.value,
                "classification": classification.value,
                "resource": resource,
                "required_role": required.value,
            },
        )


def require_same_org(user_org_id: str, resource_org_id: str) -> None:
    if user_org_id != resource_org_id:
        raise AccessDeniedError(
            "Cross-organization access denied",
            context={"user_org_id": user_org_id, "resource_org_id": resource_org_id},
        )
