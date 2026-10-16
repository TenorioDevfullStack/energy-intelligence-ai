import numpy as np
import pandas as pd
import pytest
from src.anomaly_detection import train_anomaly_model, detect_anomalies, evaluate_anomaly_model
from src.forecasting import train_forecast_models, predict_forecast, evaluate_forecast
from src.inference import EnergyPipeline
from src.train import train_project
from src.utils import DATASET_PATH
from src.data_processing import load_data


@pytest.fixture(scope="module")
def demo():
    return load_data(DATASET_PATH, require_electrical=True, require_regular=True)


@pytest.fixture(scope="module")
def trained(demo, tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    report = train_project(demo, folder, folder / "outputs", n_estimators=16)
    return EnergyPipeline.load(folder), report, folder


def test_anomaly_score_is_stable_between_batches(demo):
    bundle = train_anomaly_model(demo.iloc[:2304])
    one = detect_anomalies(demo.iloc[[2458]].reset_index(drop=True), bundle)
    batch = detect_anomalies(demo.iloc[2304:].reset_index(drop=True), bundle)
    assert one.anomaly_score.iloc[0] == pytest.approx(batch.loc[154, "anomaly_score"])
    assert batch.anomaly_score.between(0, 100).all()
    assert batch.loc[batch.is_anomaly_prediction == 0, "severity"].eq("Normal").all()


def test_supervised_metrics_only_when_labels_exist(demo):
    predicted = np.zeros(len(demo), dtype=int)
    assert evaluate_anomaly_model(demo.drop(columns="is_anomaly"), predicted) is None
    metrics = evaluate_anomaly_model(demo, predicted)
    assert metrics["false_negatives"] == 20
    assert metrics["true_negatives"] == 2860
    assert metrics["confusion_matrix"] == [[2860, 0], [20, 0]]
    assert metrics["recall"] == 0


def test_forecast_metrics():
    result = evaluate_forecast([1, 2, 3], [1, 2, 3])
    assert result == {"mae_kw": 0., "rmse_kw": 0., "r2": 1.}


def test_comparison_is_temporal_and_contains_baselines(trained):
    pipeline, report, _ = trained
    assert set(report["forecast_holdout"]) == {"forecast_v1", "forecast_v2", "baseline_daily", "baseline_profile"}
    assert report["train_end"] < report["holdout_start"]
    assert report["forecast_selection"] in report["forecast_holdout"]
    for fold in report["forecast_cv_folds"]:
        assert fold["train_end"] < fold["validation_start"]
    assert len(report["forecast_cv_folds"]) == 3
    assert report["forecast_v2_features"][-1] == "rolling_std_96"


def test_pipeline_persists_and_never_retrains(trained, demo, monkeypatch):
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    def fail(*args, **kwargs):
        raise AssertionError("Inference attempted fit")
    monkeypatch.setattr(IsolationForest, "fit", fail)
    monkeypatch.setattr(RandomForestRegressor, "fit", fail)
    pipeline, _, folder = trained
    loaded = EnergyPipeline.load(folder)
    first = pipeline.analyze(demo)
    second = loaded.analyze(demo)
    pd.testing.assert_frame_equal(first, second)
    required = {"timestamp", "active_power_kw", "expected_power_kw", "forecast_error_kw",
                "is_anomaly_prediction", "anomaly_score", "severity", "event_message"}
    assert required.issubset(first.columns)
    assert len(first) == len(demo)
    assert first.expected_power_kw.notna().all()
    selected = pipeline.forecast["selected"]
    expected_source = selected if selected in ("forecast_v1", "baseline_profile") else "forecast_v1_fallback"
    assert first.forecast_source.iloc[0] == expected_source
    assert not first.model_training_overlap.iloc[-1]
    assert first.model_training_overlap.iloc[0]


def test_inference_cannot_use_current_or_future_power(trained, demo):
    pipeline, _, _ = trained
    pipeline = EnergyPipeline(pipeline.anomaly, dict(pipeline.forecast, selected="forecast_v2"))
    raw = demo.copy()
    before = pipeline.analyze(raw).expected_power_kw
    raw.loc[2458:, "active_power_kw"] *= 10
    after = pipeline.analyze(raw).expected_power_kw
    np.testing.assert_allclose(before.iloc[:2459], after.iloc[:2459])


def test_short_history_future_and_history_overlap(trained, demo):
    pipeline, _, _ = trained
    pipeline = EnergyPipeline(pipeline.anomaly, dict(pipeline.forecast, selected="forecast_v2"))
    last = demo.iloc[[-1]]
    short = pipeline.analyze(last)
    assert short.forecast_source.iloc[0] == "forecast_v1_fallback"
    future = pipeline.forecast_next(demo)
    assert pd.Timestamp(future["timestamp"]) == demo.timestamp.iloc[-1] + pd.Timedelta(minutes=15)
    assert np.isfinite(future["predicted_power_kw"])
    with pytest.raises(ValueError, match="histórico"):
        pipeline.analyze(demo.tail(2), history=demo.tail(4))
    with pytest.raises(ValueError):
        pipeline.forecast_next(demo, demo.timestamp.iloc[-1])


def test_missing_models_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        EnergyPipeline.load(tmp_path)


def test_partial_ground_truth_never_invents_labels():
    data = pd.DataFrame({"is_anomaly": [1., np.nan, 0., np.nan]})
    metrics = evaluate_anomaly_model(data, [1, 1, 0, 1])
    assert metrics["evaluated_rows"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert evaluate_anomaly_model(pd.DataFrame({"is_anomaly":[np.nan]}), [1]) is None


@pytest.mark.parametrize("content", [b"not a joblib file", b""])
def test_corrupt_artifacts_are_actionable(tmp_path, content):
    (tmp_path / "anomaly_model.joblib").write_bytes(content)
    (tmp_path / "forecast_model.joblib").write_bytes(content)
    with pytest.raises(ValueError, match="python -m src.train"):
        EnergyPipeline.load(tmp_path)


def test_wrong_bundle_and_mixed_training_artifacts(trained, tmp_path):
    import joblib
    pipeline, _, _ = trained
    joblib.dump(0, tmp_path / "anomaly_model.joblib")
    joblib.dump(pipeline.forecast, tmp_path / "forecast_model.joblib")
    with pytest.raises(ValueError, match="python -m src.train"):
        EnergyPipeline.load(tmp_path)
    bad = dict(pipeline.forecast, train_end="2026-02-01T00:00:00")
    joblib.dump(pipeline.anomaly, tmp_path / "anomaly_model.joblib")
    joblib.dump(bad, tmp_path / "forecast_model.joblib")
    with pytest.raises(ValueError, match="python -m src.train"):
        EnergyPipeline.load(tmp_path)


@pytest.mark.parametrize("selected,source", [
    ("forecast_v1", "forecast_v1"),
    ("baseline_profile", "baseline_profile"),
    ("baseline_daily", "forecast_v1_fallback"),
    ("forecast_v2", "forecast_v1_fallback"),
])
def test_short_history_requirements_depend_on_selected_candidate(trained, demo, selected, source):
    pipeline, _, _ = trained
    bundle = dict(pipeline.forecast, selected=selected)
    result = predict_forecast(demo.tail(1).reset_index(drop=True), bundle)
    assert result.forecast_source.iloc[0] == source
    assert np.isfinite(result.expected_power_kw.iloc[0])


def test_profile_forecast_is_identical_individually_and_in_a_batch(trained, demo):
    from src.feature_engineering import typical_power
    pipeline, _, _ = trained
    bundle = dict(pipeline.forecast, selected="baseline_profile")
    one = predict_forecast(demo.tail(1).reset_index(drop=True), bundle)
    batch = predict_forecast(demo, bundle)
    expected = typical_power(demo.tail(1), bundle["profile"], day_type=True).iloc[0]
    assert one.expected_power_kw.iloc[0] == pytest.approx(expected)
    assert one.expected_power_kw.iloc[0] == pytest.approx(batch.expected_power_kw.iloc[-1])


def test_daily_baseline_needs_only_the_reading_twenty_four_hours_earlier(trained, demo):
    pipeline, _, _ = trained
    bundle = dict(pipeline.forecast, selected="baseline_daily")
    rows = demo.tail(97).reset_index(drop=True)
    result = predict_forecast(rows, bundle)
    assert result.forecast_source.iloc[-1] == "baseline_daily"
    assert result.expected_power_kw.iloc[-1] == rows.active_power_kw.iloc[0]
    missing = predict_forecast(rows.iloc[1:].reset_index(drop=True), bundle)
    assert missing.forecast_source.iloc[-1] == "forecast_v1_fallback"


@pytest.mark.parametrize("selected", ["forecast_v1", "forecast_v2", "baseline_daily", "baseline_profile"])
def test_selected_candidate_matches_prediction_with_complete_history(trained, demo, selected):
    from src.feature_engineering import build_forecast_features
    from src.forecasting import candidate_predictions
    pipeline, _, _ = trained
    bundle = dict(pipeline.forecast, selected=selected)
    features = build_forecast_features(demo)
    expected = candidate_predictions(features.tail(1), demo.tail(1), bundle)[selected][0]
    result = predict_forecast(demo, bundle)
    assert result.forecast_source.iloc[-1] == selected
    assert result.expected_power_kw.iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("selected,missing_lag,source", [
    ("baseline_daily", 4, "baseline_daily"),
    ("baseline_daily", 96, "forecast_v1_fallback"),
    ("forecast_v2", 672, "forecast_v1_fallback"),
])
def test_missing_readings_affect_only_candidates_that_require_them(trained, demo, selected, missing_lag, source):
    pipeline, _, _ = trained
    bundle = dict(pipeline.forecast, selected=selected)
    rows = demo.tail(700).reset_index(drop=True)
    last = len(rows) - 1
    expected_daily = rows.active_power_kw.iloc[last - 96]
    sparse = rows.drop(index=last - missing_lag).reset_index(drop=True)
    result = predict_forecast(sparse, bundle)
    assert result.forecast_source.iloc[-1] == source
    if source == "baseline_daily":
        assert result.expected_power_kw.iloc[-1] == expected_daily
