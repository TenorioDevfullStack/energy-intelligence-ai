"""A fresh public checkout must render without manually generated model files."""
import io
import json
import copy
from unittest.mock import patch

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


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_bundled_demo_preparation_uses_installed_file_bytes(tmp_path, monkeypatch, line_ending):
    import app.resources as resources
    import src.utils

    content = DATASET_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", line_ending)
    bundled = tmp_path / "energy_consumption_30days.csv"
    bundled.write_bytes(content)
    monkeypatch.setattr(src.utils, "DATASET_PATH", bundled)
    resources.load_demo_resources.clear()
    demo = resources.load_demo_resources(content)
    assert demo.prepared_demo
    assert demo.metrics["anomaly_threshold_calibration"]["selected_threshold"] == 68


def test_one_byte_change_to_bundled_demo_is_rejected_before_training():
    from app.resources import load_demo_resources

    content = DATASET_PATH.read_bytes()
    changed = bytes([content[0] ^ 1]) + content[1:]
    assert len(changed) == len(content)
    with patch("src.train.train_project", side_effect=AssertionError("Must reject before training")) as train:
        with pytest.raises(ValueError, match="demo"):
            load_demo_resources(changed)
    train.assert_not_called()


def test_valid_upload_only_runs_inference_after_demo_startup(tmp_path, monkeypatch):
    import app.resources as resources
    import src.utils

    content = DATASET_PATH.read_bytes()
    resources.load_dashboard_resources(str(tmp_path), resources.resource_signature(tmp_path), content)
    monkeypatch.setattr(src.utils, "MODEL_DIR", tmp_path)
    upload = io.BytesIO(b"timestamp,active_power_kw,voltage_v,power_factor\n2026-02-01 18:15:00,50,127,.94\n")
    with patch("streamlit.file_uploader", return_value=upload), \
            patch.object(resources, "load_demo_resources", side_effect=AssertionError("Upload must reuse prepared demo")) as prepare, \
            patch("src.train.train_project", side_effect=AssertionError("Upload must not train")) as train, \
            patch.object(IsolationForest, "fit", side_effect=AssertionError("Upload must not fit")), \
            patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Upload must not fit")):
        app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception and not app.error
    assert any(metric.label == "Pontos anômalos" and metric.value == "1" for metric in app.metric)
    prepare.assert_not_called()
    train.assert_not_called()


def test_invalid_upload_is_rejected_before_demo_preparation(tmp_path, monkeypatch):
    import app.resources as resources
    import src.utils

    monkeypatch.setattr(src.utils, "MODEL_DIR", tmp_path)
    with patch("streamlit.file_uploader", return_value=io.BytesIO(b"not,a,valid,csv\n")), \
            patch.object(resources, "load_demo_resources", side_effect=AssertionError("Invalid upload must not prepare")) as prepare, \
            patch("src.train.train_project", side_effect=AssertionError("Invalid upload must not train")) as train, \
            patch.object(IsolationForest, "fit", side_effect=AssertionError("Invalid upload must not fit")), \
            patch.object(RandomForestRegressor, "fit", side_effect=AssertionError("Invalid upload must not fit")):
        app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert any("CSV incompatível" in item.value for item in app.error)
    prepare.assert_not_called()
    train.assert_not_called()


def test_persisted_models_avoid_automatic_demo_preparation(tmp_path):
    import app.resources as resources

    content = DATASET_PATH.read_bytes()
    demo = resources.load_demo_resources(content)
    demo.pipeline.save(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps(demo.metrics), encoding="utf-8")
    with patch.object(resources, "load_demo_resources", side_effect=AssertionError("Models are already persisted")) as prepare, \
            patch("src.train.train_project", side_effect=AssertionError("Must load persisted models")) as train:
        loaded = resources.load_dashboard_resources(str(tmp_path), resources.resource_signature(tmp_path), content)
    assert not loaded.prepared_demo
    assert loaded.pipeline.anomaly["threshold"] == 68
    prepare.assert_not_called()
    train.assert_not_called()
