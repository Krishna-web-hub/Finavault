from unittest.mock import MagicMock

from finvault.agents.orchestrator import Orchestrator
from finvault.config import settings
from finvault.models import Role, User


def test_per_agent_models_default_to_global(monkeypatch):
    monkeypatch.setattr(settings, "finvault_model", "default-global-model")
    monkeypatch.setattr(settings, "finvault_orchestrator_model", None)
    monkeypatch.setattr(settings, "finvault_retriever_model", None)
    monkeypatch.setattr(settings, "finvault_analyst_model", None)
    monkeypatch.setattr(settings, "finvault_compliance_model", None)

    user = User(username="test_user", role=Role.ANALYST, org_id="org1")
    retriever = MagicMock()
    audit_log = MagicMock()

    orchestrator = Orchestrator(retriever=retriever, user=user, audit_log=audit_log)

    assert orchestrator._agent._model == "default-global-model"
    assert orchestrator._retriever_agent._agent._model == "default-global-model"
    assert orchestrator._analyst_agent._agent._model == "default-global-model"
    assert orchestrator._compliance_agent._model == "default-global-model"


def test_per_agent_models_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "finvault_model", "default-global-model")
    monkeypatch.setattr(settings, "finvault_orchestrator_model", "kimi-k3")
    monkeypatch.setattr(settings, "finvault_retriever_model", "kimi-k2.7-code-highspeed")
    monkeypatch.setattr(settings, "finvault_analyst_model", "kimi-k3")
    monkeypatch.setattr(settings, "finvault_compliance_model", "kimi-k2.7-code-highspeed")

    user = User(username="test_user", role=Role.ANALYST, org_id="org1")
    retriever = MagicMock()
    audit_log = MagicMock()

    orchestrator = Orchestrator(retriever=retriever, user=user, audit_log=audit_log)

    assert orchestrator._agent._model == "kimi-k3"
    assert orchestrator._retriever_agent._agent._model == "kimi-k2.7-code-highspeed"
    assert orchestrator._analyst_agent._agent._model == "kimi-k3"
    assert orchestrator._compliance_agent._model == "kimi-k2.7-code-highspeed"


def test_per_agent_models_constructor_overrides():
    user = User(username="test_user", role=Role.ANALYST, org_id="org1")
    retriever = MagicMock()
    audit_log = MagicMock()

    orchestrator = Orchestrator(
        retriever=retriever,
        user=user,
        audit_log=audit_log,
        orchestrator_model="custom-orch",
        retriever_model="custom-retriever",
        analyst_model="custom-analyst",
        compliance_model="custom-compliance",
    )

    assert orchestrator._agent._model == "custom-orch"
    assert orchestrator._retriever_agent._agent._model == "custom-retriever"
    assert orchestrator._analyst_agent._agent._model == "custom-analyst"
    assert orchestrator._compliance_agent._model == "custom-compliance"


def test_legacy_model_param_compatibility():
    user = User(username="test_user", role=Role.ANALYST, org_id="org1")
    retriever = MagicMock()
    audit_log = MagicMock()

    orchestrator = Orchestrator(
        retriever=retriever,
        user=user,
        audit_log=audit_log,
        model="legacy-all-model",
    )

    assert orchestrator._agent._model == "legacy-all-model"
    assert orchestrator._retriever_agent._agent._model == "legacy-all-model"
    assert orchestrator._analyst_agent._agent._model == "legacy-all-model"
    assert orchestrator._compliance_agent._model == "legacy-all-model"
