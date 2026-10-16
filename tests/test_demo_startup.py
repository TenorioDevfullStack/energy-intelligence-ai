"""A fresh public checkout must render without manually generated model files."""
import io
import json
import copy

import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from streamlit.testing.v1 import AppTest

from src.data_processing import load_data
from src.utils import DATASET_PATH, ROOT


@pytest.mark.parametrize("corrupt", [False, True])
def test_dashboard_without_valid_artifacts_prepares_demo_once(tmp_path, monkeypatch, corrupt):
    import src.utils

    if corrupt:
        (tmp_path / "anomaly_model.joblib").write_bytes(b"corrupt")
        (tmp_path / "forecast_model.joblib").write_bytes(b"corrupt")
    monkeypatch.setattr(src.utils, "MODEL_DIR", tmp_path)
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=180).run()
    assert not app.exception and not app.error
    assert len(app.metric) >= 10
    assert any("demo" in message.value.lower() for message in app.info)
    assert any("Forecast" in item.value for item in app.subheader)

    def no_fit(*args, **kwargs):
        raise AssertionError("Reruns and inference must reuse prepared demo models")

    monkeypatch.setattr(IsolationForest, "fit", no_fit)
    monkeypatch.setattr(RandomForestRegressor, "fit", no_fit)
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception and not app.error
    assert len(app.metric) >= 10


def test_demo_upload_analysis_and_invalid_csv_contract():
    from app.resources import load_demo_resources

    resources = load_demo_resources(DATASET_PATH.read_bytes())
    rows = pd.DataFrame({"timestamp": pd.date_range("2026-02-01T18:15", periods=3, freq="15min"),
                         "active_power_kw": 50., "voltage_v": 127., "power_factor": .94})
    for labelled in (False, True):
        sample = rows.assign(is_anomaly=1) if labelled else rows
        uploaded = load_data(io.BytesIO(sample.to_csv(index=False).encode()), require_electrical=True)
        result = resources.pipeline.analyze_with_events(uploaded)
        assert len(result["results"]) == 3
        assert result["summary"]["confirmed_events"] == 1
        assert result["events"].iloc[0].duration_minutes == 30

    for invalid in (b"", b"timestamp,active_power_kw\n2026-02-01,5\n",
                    b"timestamp,active_power_kw,voltage_v,power_factor\nbad,5,127,.94\n"):
        with pytest.raises(ValueError):
            load_data(io.BytesIO(invalid), require_electrical=True)


def test_automatic_preparation_rejects_unknown_dataset():
    from app.resources import load_demo_resources

    with pytest.raises(ValueError, match="demo"):
        load_demo_resources(b"timestamp,active_power_kw\n2026-01-01,2\n")


@pytest.mark.parametrize("report", [None, "broken json", '{"train_end": "wrong"}'])
def test_dashboard_missing_or_corrupt_metrics_uses_explicit_demo(tmp_path, report):
    from app.resources import load_demo_resources, load_dashboard_resources, resource_signature

    demo_content = DATASET_PATH.read_bytes()
    original = load_demo_resources(demo_content)
    original.pipeline.save(tmp_path)
    if report is not None:
        (tmp_path / "metrics.json").write_text(report, encoding="utf-8")
    result = load_dashboard_resources(str(tmp_path), resource_signature(tmp_path), demo_content)
    assert result.prepared_demo
    assert result.metrics["anomaly_threshold_calibration"]["selected_threshold"] == 68


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(anomaly_comparison={}),
    lambda m: m.update(forecast_cv=None),
    lambda m: m["anomaly_threshold_calibration"].pop("default_threshold"),
    lambda m: m["anomaly_comparison"]["holdout"]["calibrated"].pop("recall"),
    lambda m: m["anomaly_threshold_calibration"].update(default_threshold="47.2"),
    lambda m: m["anomaly_holdout"].update(precision="1.0"),
])
def test_structurally_corrupt_metrics_fall_back_before_rendering(tmp_path, mutation):
    from app.resources import load_demo_resources, load_dashboard_resources, resource_signature

    content = DATASET_PATH.read_bytes()
    demo = load_demo_resources(content)
    demo.pipeline.save(tmp_path)
    report = copy.deepcopy(demo.metrics)
    mutation(report)
    (tmp_path / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    loaded = load_dashboard_resources(str(tmp_path), resource_signature(tmp_path), content)
    assert loaded.prepared_demo
