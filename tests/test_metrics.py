"""Tests for finvault/metrics.py and the /metrics endpoint.

The property that matters most here is one you cannot see by reading a
dashboard: **label cardinality**. Prometheus stores one time series per
distinct label combination, so a label carrying a document id or a raw URL
path grows without bound and eventually takes the metrics backend down. The
tests below pin the two places this codebase could get that wrong.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from finvault.api.error_handlers import install_error_handlers
from finvault.errors import AccessDeniedError, AgentExecutionError
from finvault.metrics import observe_duration, record_error, record_tokens


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/documents/{document_id}")
    def get_document(document_id: str) -> dict:
        return {"id": document_id}

    @app.get("/denied")
    def denied() -> dict:
        raise AccessDeniedError("no clearance")

    @app.get("/bug")
    def bug() -> dict:
        raise RuntimeError("boom")

    from finvault.metrics import render

    @app.get("/metrics")
    def metrics():
        from fastapi import Response

        body, content_type = render()
        return Response(content=body, media_type=content_type)

    return TestClient(app, raise_server_exceptions=False)


def _sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_a_path_parameter_never_becomes_a_label(client: TestClient) -> None:
    """The route *template*, not the concrete path. One series per document
    id is the classic way to melt a Prometheus server."""
    for document_id in ("doc-1", "doc-2", "doc-3"):
        client.get(f"/documents/{document_id}")

    body = client.get("/metrics").text
    assert 'route="/documents/{document_id}"' in body
    assert "doc-1" not in body
    assert "doc-2" not in body


def test_unmatched_paths_collapse_to_one_series(client: TestClient) -> None:
    """Probe traffic scanning for admin panels would otherwise mint a series
    per URL an attacker invents."""
    for path in ("/wp-admin", "/.env", "/phpmyadmin"):
        client.get(path)

    body = client.get("/metrics").text
    assert 'route="unmatched"' in body
    assert "wp-admin" not in body


def test_requests_are_counted_by_status_class_not_status_code(client: TestClient) -> None:
    before = _sample(
        "finvault_http_requests_total",
        {"method": "GET", "route": "/documents/{document_id}", "status_class": "2xx"},
    )
    client.get("/documents/abc")
    after = _sample(
        "finvault_http_requests_total",
        {"method": "GET", "route": "/documents/{document_id}", "status_class": "2xx"},
    )
    assert after == before + 1


def test_errors_are_split_into_expected_refusals_and_incidents(client: TestClient) -> None:
    """This split is what makes an alert rule possible: paging on a routine
    403 trains people to ignore the pager."""
    before_expected = _sample("finvault_errors_total", {"error_code": "access_denied", "branch": "expected"})
    before_incident = _sample("finvault_errors_total", {"error_code": "internal_error", "branch": "incident"})

    client.get("/denied")
    client.get("/bug")

    assert (
        _sample("finvault_errors_total", {"error_code": "access_denied", "branch": "expected"}) == before_expected + 1
    )
    assert (
        _sample("finvault_errors_total", {"error_code": "internal_error", "branch": "incident"}) == before_incident + 1
    )


def test_a_dependency_failure_counts_as_an_incident() -> None:
    before = _sample("finvault_errors_total", {"error_code": "agent_execution_failed", "branch": "incident"})
    record_error(AgentExecutionError.code, branch="incident")
    assert (
        _sample("finvault_errors_total", {"error_code": "agent_execution_failed", "branch": "incident"}) == before + 1
    )


def test_tokens_are_recorded_by_direction() -> None:
    labels = {"agent": "metrics-test", "model": "m", "direction": "input"}
    before = _sample("finvault_llm_tokens_total", labels)
    record_tokens(agent="metrics-test", model="m", input_tokens=100, output_tokens=50)
    assert _sample("finvault_llm_tokens_total", labels) == before + 100
    assert _sample("finvault_llm_tokens_total", {**labels, "direction": "output"}) == 50


def test_spend_stays_zero_while_rates_are_unconfigured(monkeypatch) -> None:
    """A flat-zero cost panel is the honest signal for "not configured". A
    guessed rate on a cost dashboard is worse than an obvious zero."""
    from finvault.config import settings

    monkeypatch.setattr(settings, "finvault_price_input_per_million", 0.0)
    monkeypatch.setattr(settings, "finvault_price_output_per_million", 0.0)
    record_tokens(agent="pricing-off", model="m", input_tokens=1_000_000, output_tokens=1_000_000)
    assert _sample("finvault_llm_cost_usd_total", {"agent": "pricing-off", "model": "m"}) == 0.0


def test_configured_rates_turn_tokens_into_spend(monkeypatch) -> None:
    from finvault.config import settings

    monkeypatch.setattr(settings, "finvault_price_input_per_million", 3.0)
    monkeypatch.setattr(settings, "finvault_price_output_per_million", 15.0)
    record_tokens(agent="pricing-on", model="m", input_tokens=1_000_000, output_tokens=1_000_000)
    assert _sample("finvault_llm_cost_usd_total", {"agent": "pricing-on", "model": "m"}) == pytest.approx(18.0)


def test_a_failing_block_is_still_timed() -> None:
    """Failures are usually the slow ones — a timeout, a retry chain — so
    dropping them would make latency look best exactly when it is worst."""
    from finvault.metrics import agent_duration_seconds

    def count() -> float:
        return REGISTRY.get_sample_value("finvault_agent_duration_seconds_count", {"agent": "timed-failure"}) or 0.0

    before = count()
    with pytest.raises(ValueError), observe_duration(agent_duration_seconds, agent="timed-failure"):
        raise ValueError("boom")
    assert count() == before + 1


def test_the_metrics_endpoint_serves_prometheus_exposition(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP finvault_http_requests_total" in response.text
    assert "# TYPE finvault_http_requests_total counter" in response.text


def test_metrics_expose_no_identity_or_content(client: TestClient) -> None:
    """The endpoint is unauthenticated by design, so what it cannot contain
    is the whole safety argument."""
    client.get("/documents/secret-doc-id")
    client.get("/denied")
    body = client.get("/metrics").text

    assert "secret-doc-id" not in body
    for forbidden in ("org_id", "actor", "user_id", "question", "answer", "request_id"):
        assert f'{forbidden}="' not in body
