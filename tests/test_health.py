"""Liveness route used by Docker HEALTHCHECK and compose."""

from fastapi.testclient import TestClient

from web.backend.app import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "tradingcrew"
