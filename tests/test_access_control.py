from __future__ import annotations

import pytest

from finvault.models import Classification, Role
from finvault.security.access_control import AccessDeniedError, check_clearance, require_clearance, require_same_org


@pytest.mark.parametrize(
    "role,classification,expected",
    [
        (Role.VIEWER, Classification.PUBLIC, True),
        (Role.VIEWER, Classification.INTERNAL, True),
        (Role.VIEWER, Classification.CONFIDENTIAL, False),
        (Role.VIEWER, Classification.RESTRICTED, False),
        (Role.ANALYST, Classification.CONFIDENTIAL, True),
        (Role.ANALYST, Classification.RESTRICTED, False),
        (Role.COMPLIANCE_OFFICER, Classification.RESTRICTED, True),
        (Role.ADMIN, Classification.RESTRICTED, True),
    ],
)
def test_check_clearance_matrix(role: Role, classification: Classification, expected: bool) -> None:
    assert check_clearance(role, classification) is expected


def test_require_clearance_raises_when_denied() -> None:
    with pytest.raises(AccessDeniedError):
        require_clearance(Role.VIEWER, Classification.RESTRICTED)


def test_require_clearance_passes_when_allowed() -> None:
    require_clearance(Role.ADMIN, Classification.RESTRICTED)  # must not raise


def test_require_same_org_raises_on_mismatch() -> None:
    with pytest.raises(AccessDeniedError):
        require_same_org("org-a", "org-b")


def test_require_same_org_passes_when_matching() -> None:
    require_same_org("org-a", "org-a")  # must not raise
