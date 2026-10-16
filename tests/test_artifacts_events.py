import copy
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.api import create_app

from src.anomaly_detection import train_anomaly_model
from src.data_processing import load_data
from src.inference import EnergyPipeline
from src.train import train_project
from src.utils import DATASET_PATH


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("stage2-artifacts")
    raw = load_data(DATASET_PATH, require_electrical=True)
    train_project(raw, folder, folder / "outputs", n_estimators=8, min_event_points=4)
    return EnergyPipeline.load(folder), folder, raw


@pytest.mark.parametrize("mutation", [
    lambda a, f: a.update(threshold_calibration=None),
    lambda a, f: a.pop("threshold_calibration"),
    lambda a, f: (a.pop("training_data_sha256"), f.pop("training_data_sha256")),
    lambda a, f: f.update(training_data_sha256="different dataset"),
    lambda a, f: a["threshold_calibration"].update(status="unknown"),
    lambda a, f: a["threshold_calibration"].update(status="no_labels"),
])
def test_stage2_artifacts_require_complete_consistent_calibration(artifacts, mutation):
    pipeline, _, _ = artifacts
    a, f = copy.deepcopy(pipeline.anomaly), copy.deepcopy(pipeline.forecast)
    mutation(a, f)
    with pytest.raises(ValueError):
        EnergyPipeline._validate_artifacts(a, f)


def test_true_legacy_artifacts_still_load_without_calibration(artifacts, tmp_path):
    pipeline, _, raw = artifacts
    forecast = dict(pipeline.forecast)
    forecast.pop("anomaly_train_end", None)
    forecast.pop("training_data_sha256", None)
    legacy = EnergyPipeline(train_anomaly_model(raw.iloc[:2304]), forecast)
    legacy.save(tmp_path)
    loaded = EnergyPipeline.load(tmp_path)
    assert loaded.analyze(raw.tail(1)).is_anomaly_prediction.iloc[0] in (0, 1)


def test_incomplete_calibration_reports_not_ready_instead_of_crashing(artifacts, tmp_path):
    pipeline, _, _ = artifacts
    bad = EnergyPipeline(dict(pipeline.anomaly, threshold_calibration=None), pipeline.forecast)
    bad.save(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 503
        row = {"timestamp": "2026-02-01T12:00:00", "active_power_kw": 5., "voltage_v": 127., "power_factor": .94}
        assert client.post("/analyze/events", json={"records": [row]}).status_code == 503


def test_custom_event_policy_survives_persistence_and_api_defaults(artifacts):
    pipeline, folder, _ = artifacts
    rows = pd.DataFrame({"timestamp": pd.date_range("2026-02-01T18:15", periods=3, freq="15min"),
                         "active_power_kw": 50., "voltage_v": 127., "power_factor": .94})
    events = pipeline.analyze_with_events(rows)["events"]
    assert events.iloc[0].status == "suspected"
    with TestClient(create_app(folder)) as client:
        records = rows.assign(timestamp=rows.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")).to_dict(orient="records")
        response = client.post("/analyze/events", json={"records": records})
        assert response.status_code == 200
        assert response.json()["events"][0]["status"] == "suspected"
        response = client.post("/analyze/events", json={"records": records, "min_event_points": 2})
        assert response.json()["events"][0]["status"] == "confirmed"


@pytest.mark.parametrize("points,gap", [(1500, 15.), (2, .5), (2, 2880.)])
def test_dashboard_accepts_supported_persisted_event_policies(artifacts, tmp_path, monkeypatch, points, gap):
    import src.utils
    from streamlit.testing.v1 import AppTest
    pipeline, folder, _ = artifacts
    modified = EnergyPipeline(dict(pipeline.anomaly, event_policy={"min_event_points": points, "max_gap_minutes": gap}), pipeline.forecast)
    modified.save(tmp_path)
    (tmp_path / "metrics.json").write_bytes((folder / "metrics.json").read_bytes())
    dataset = tmp_path / "healthy.csv"
    pd.DataFrame({"timestamp": ["2026-02-01T12:00:00"], "active_power_kw": [5.],
                  "voltage_v": [127.], "power_factor": [.94]}).to_csv(dataset, index=False)
    monkeypatch.setattr(src.utils, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(src.utils, "DATASET_PATH", dataset)
    app = AppTest.from_file(str(src.utils.ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception and not app.error
    assert app.sidebar.number_input[0].value == points
    assert app.sidebar.number_input[1].value == gap
