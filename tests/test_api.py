import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from src.inference import EnergyPipeline
from src.train import train_project
from src.data_processing import load_data
from src.utils import DATASET_PATH


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("api-models")
    train_project(load_data(DATASET_PATH, require_electrical=True), folder, folder / "outputs", n_estimators=8)
    with TestClient(create_app(folder)) as instance:
        yield instance


def reading(**changes):
    row = {"timestamp": "2026-02-01T12:00:00", "active_power_kw": 5., "voltage_v": 127., "power_factor": .94}
    return dict(row, **changes)


def test_health_and_analysis(client):
    assert client.get("/health").json()["models_ready"] is True
    response = client.post("/analyze", json={"records": [reading()]})
    assert response.status_code == 200
    row = response.json()["results"][0]
    assert 0 <= row["anomaly_score"] <= 100
    assert row["predicted_power_kw"] == row["expected_power_kw"]
    assert isinstance(row["anomaly"], bool)
    assert row["forecast_source"] == client.get("/health").json()["forecast_model"]
    assert response.json()["anomaly_metrics"] is None


def test_anomaly_and_future_forecast(client):
    response = client.post("/predict/anomaly", json={"records": [reading(active_power_kw=50)]})
    assert response.status_code == 200
    assert response.json()["results"][0]["anomaly"]
    future = client.post("/predict/forecast", json={"history": [
        {"timestamp": "2026-02-01T12:00:00", "active_power_kw": 5.}]})
    assert future.status_code == 200
    assert future.json()["timestamp"] == "2026-02-01T12:15:00"


@pytest.mark.parametrize("payload", [{"records": []}, {"records": [reading(power_factor=2)]},
    {"records": [reading(active_power_kw=-1)]}, {"records": [reading(timestamp="bad")]},
    {"records": [reading(timestamp="2026-02-01T12:01:00")]},
    {"records": [reading(), reading()]}, {"records": [{"timestamp":"2026-02-01T12:00:00","active_power_kw":5}]},
    {"records": [reading(unknown=42)]}, {"records": [reading(timestamp="2026-02-01T12:00:00Z")]}])
def test_invalid_input_returns_422(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422


def test_history_must_precede_measurement(client):
    payload = {"records": [reading()], "history": [{"timestamp": "2026-02-02T12:00:00", "active_power_kw":5}]}
    assert client.post("/analyze", json=payload).status_code == 422


def test_missing_models_have_readiness_status(tmp_path):
    with TestClient(create_app(tmp_path)) as instance:
        response = instance.get("/health")
        assert response.status_code == 503
        assert response.json()["models_ready"] is False
        assert instance.post("/analyze", json={"records": [reading()]}).status_code == 503


def test_partial_optional_fields_are_valid(client):
    payload = {"records": [reading(current_a=10, is_anomaly=0),
        reading(timestamp="2026-02-01T12:15:00")]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["anomaly_metrics"]["evaluated_rows"] == 1


def test_corrupt_models_return_503(tmp_path):
    for name in ("anomaly_model.joblib", "forecast_model.joblib"):
        (tmp_path / name).write_bytes(b"not a joblib file")
    with TestClient(create_app(tmp_path)) as instance:
        assert instance.get("/health").status_code == 503
        assert instance.post("/analyze", json={"records": [reading()]}).status_code == 503


def test_short_api_requests_use_the_selected_candidate_when_history_is_not_required(client):
    selected = client.get("/health").json()["forecast_model"]
    assert selected == "baseline_profile"
    analyzed = client.post("/analyze", json={"records": [reading()]}).json()["results"][0]
    forecast = client.post("/predict/forecast", json={"history": [
        {"timestamp": "2026-02-01T12:00:00", "active_power_kw": 5.}]}).json()
    assert analyzed["forecast_source"] == selected
    assert forecast["forecast_source"] == selected


def test_events_endpoint_groups_points_and_keeps_point_api(client):
    payload = {"records": [reading(timestamp=f"2026-02-01T18:{minute}:00", active_power_kw=50.)
                            for minute in ("15", "30", "45")]}
    response = client.post("/analyze/events", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_events"] == body["summary"]["confirmed_events"] == 1
    assert body["events"][0]["duration_minutes"] == 30
    assert body["events"][0]["number_of_points"] == 3
    assert body["events"][0]["status"] == "confirmed"
    assert client.post("/analyze", json=payload).json()["results"][0]["anomaly"]
    payload["min_event_points"] = 4
    assert client.post("/analyze/events", json=payload).json()["events"][0]["status"] == "suspected"


def test_events_endpoint_supports_empty_events(client):
    response = client.post("/analyze/events", json={"records": [reading()]})
    assert response.status_code == 200
    assert response.json()["events"] == []
    assert response.json()["summary"]["total_events"] == 0


@pytest.mark.parametrize("options", [{"min_event_points": 0}, {"min_event_points": 1.5},
                                     {"max_gap_minutes": 0}, {"unknown": 1}, {"max_gap_minutes": 1e308}])
def test_events_endpoint_validates_policy(client, options):
    assert client.post("/analyze/events", json={"records": [reading()], **options}).status_code == 422


def test_events_endpoint_reports_unavailable_models(tmp_path):
    with TestClient(create_app(tmp_path)) as instance:
        assert instance.post("/analyze/events", json={"records": [reading()]}).status_code == 503
