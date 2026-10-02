from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from irrigaiq_ml.api import app
from irrigaiq_ml.simulation import inject, normal_flow


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def field(soil: float, phase: str = "DEVELOPMENT", temp: float = 31) -> dict:
    return {
        "fieldId": f"f-{soil}",
        "soilMoisturePct": soil,
        "temperatureC": temp,
        "windSpeedKmh": 14,
        "rainfallMmAnnualised": 600,
        "growthPhase": phase,
        "mulched": False,
    }


def test_health_lists_loaded_models(client):
    body = client.get("/health").json()
    assert body["status"] == "UP"
    assert {"irrigation", "equipment"} <= set(body["models"])


def test_dry_soil_needs_water_and_wet_soil_late_in_season_does_not(client):
    response = client.post("/v1/irrigation/predict", json={"fields": [field(20), field(55, "LATE_SEASON", 22)]})
    assert response.status_code == 200
    dry, wet = response.json()["predictions"]
    assert dry["irrigationRequired"] is True and dry["need"] == "High"
    assert wet["irrigationRequired"] is False
    assert any(d["feature"] == "Soil_Moisture" for d in dry["drivers"])
    assert abs(sum(dry["probabilities"].values()) - 1) < 1e-6


def test_unknown_growth_phase_is_rejected(client):
    response = client.post("/v1/irrigation/predict", json={"fields": [field(30, "FLOWERING_ISH")]})
    assert response.status_code == 422


def test_burst_is_flagged_with_hedged_causes(client):
    series = normal_flow(datetime(2026, 8, 1), 22, rate_lph=1500, seed=3)
    cut = 21 * 24
    series = inject(series, "burst", cut + 10, 4, rate_lph=1500)

    def points(ps):
        return [{"at": p.at.isoformat(), "litres": p.litres, "scheduled": p.scheduled} for p in ps]

    body = client.post(
        "/v1/anomaly/water", json={"meterId": "m1", "history": points(series[:cut]), "recent": points(series[cut : cut + 14])}
    ).json()
    assert body["status"] == "OK"
    assert body["anomaly"]["direction"] == "HIGH"
    assert body["anomaly"]["severity"] in {"MEDIUM", "HIGH"}
    assert "Possible" in body["anomaly"]["headline"]


def test_short_history_is_reported_not_guessed(client):
    now = datetime(2026, 8, 1)
    point = {"at": now.isoformat(), "litres": 10, "scheduled": False}
    history = [{**point, "at": (now - timedelta(hours=h)).isoformat()} for h in range(1, 10)]
    body = client.post("/v1/anomaly/water", json={"meterId": "m1", "history": history, "recent": [point]}).json()
    assert body["status"] == "INSUFFICIENT_HISTORY"


def test_worn_pump_is_riskier_than_a_healthy_one(client):
    healthy = {"energyRatio": 1.02, "flowRatio": 0.99, "vibrationMmS": 1.6, "serviceOverdueRatio": 0.3, "runtimeHPerDay": 7}
    worn = {"energyRatio": 1.3, "energyTrend": 0.06, "flowRatio": 0.78, "vibrationMmS": 6.5, "serviceOverdueRatio": 1.6, "runtimeHPerDay": 9}
    body = client.post(
        "/v1/equipment/risk",
        json={"assets": [{"assetId": "a", "readings": healthy}, {"assetId": "b", "readings": worn}]},
    ).json()
    a, b = body["assets"]
    assert b["failureProbability"] > a["failureProbability"]
    assert b["riskLevel"] in {"HIGH", "CRITICAL"}
    assert b["indicators"], "a risky pump must say why"
    assert "Motor temperature" in a["missing"]


def test_service_token_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("IRRIGAIQ_ML_TOKEN", "s3cret")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-Service-Token": "s3cret"}).status_code == 200
