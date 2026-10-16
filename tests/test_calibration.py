import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import detect_anomalies, evaluate_anomaly_model, train_anomaly_model
from src.data_processing import load_data
from src.inference import EnergyPipeline
from src.train import train_project
from src.utils import DATASET_PATH


def validation(scores=(10, 50, 70, 90), labels=(0, 0, 1, 1)):
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-02", periods=len(scores), freq="15min"),
                         "anomaly_score": scores, "is_anomaly": labels})


def calibrate(rows, **kwargs):
    from src.anomaly_calibration import calibrate_anomaly_threshold
    return calibrate_anomaly_threshold(rows, train_end="2026-01-01T23:45:00",
                                      holdout_start="2026-01-03T00:00:00", **kwargs)


def test_threshold_maximizes_f1_under_recall_constraint():
    threshold, table, metadata = calibrate(validation(), thresholds=(0, 50, 60, 80), min_recall=.8)
    assert threshold == 60
    assert metadata["status"] == "calibrated"
    selected = table.loc[table.threshold.eq(threshold)].iloc[0]
    assert selected.f1 == 1 and selected.recall == 1 and selected.false_positives == 0
    assert {"precision", "recall", "f1", "false_positive_rate", "false_positives",
            "false_negatives", "true_positives", "true_negatives"}.issubset(table)


def test_threshold_ties_prefer_fewer_false_positives_then_higher_cut():
    threshold, _, _ = calibrate(validation((10, 60, 65, 30, 90), (0, 0, 0, 1, 1)),
                                 thresholds=(20, 25, 70), min_recall=.5)
    assert threshold == 70  # F1=2/3 for all three; threshold 70 has no FP.
    threshold, _, _ = calibrate(validation(), thresholds=(55, 60), min_recall=.8)
    assert threshold == 60


@pytest.mark.parametrize("date", ["2026-01-01T23:45:00", "2026-01-03T00:00:00"])
def test_calibration_rejects_training_or_holdout_rows(date):
    rows = validation()
    rows.loc[0, "timestamp"] = pd.Timestamp(date)
    with pytest.raises(ValueError, match="validação"):
        calibrate(rows)


@pytest.mark.parametrize("rows,reason", [
    (validation().drop(columns="is_anomaly"), "no_labels"),
    (validation(labels=(np.nan,) * 4), "no_labels"),
    (validation(labels=(0, 0, 0, 0)), "insufficient_classes"),
    (validation(labels=(1, 1, 1, 1)), "insufficient_classes"),
])
def test_unavailable_calibration_keeps_default_detector(rows, reason):
    threshold, table, metadata = calibrate(rows)
    assert threshold is None
    assert metadata["status"] == reason
    assert table.empty


def test_partial_labels_and_no_feasible_threshold():
    threshold, _, metadata = calibrate(validation(labels=(0, np.nan, 1, 1)), thresholds=(60,))
    assert threshold == 60 and metadata["labelled_rows"] == 3
    threshold, _, metadata = calibrate(validation(), thresholds=(100,))
    assert threshold is None and metadata["status"] == "no_feasible_threshold"


@pytest.mark.parametrize("kwargs", [{"min_recall": 2}, {"thresholds": ()},
                                    {"thresholds": (np.nan,)}, {"thresholds": (-1,)}])
def test_invalid_calibration_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        calibrate(validation(), **kwargs)


def test_detector_uses_persisted_threshold_and_normal_severity():
    demo = load_data(DATASET_PATH, require_electrical=True)
    bundle = train_anomaly_model(demo.iloc[:1728])
    original = detect_anomalies(demo.tail(576), bundle)
    raised = detect_anomalies(demo.tail(576), dict(bundle, threshold=100.))
    assert raised.is_anomaly_prediction.equals(raised.anomaly_score.ge(100).astype(int))
    assert raised.is_anomaly_prediction.sum() < original.is_anomaly_prediction.sum()
    assert raised.loc[raised.is_anomaly_prediction.eq(0), "severity"].eq("Normal").all()
    pd.testing.assert_series_equal(original.anomaly_score, raised.anomaly_score)


def test_additional_metrics_use_confusion_counts():
    metrics = evaluate_anomaly_model(pd.DataFrame({"is_anomaly": [0, 0, 1, 1]}), [1, 0, 1, 0])
    assert metrics["false_positive_rate"] == .5
    assert metrics["specificity"] == .5
    assert metrics["balanced_accuracy"] == .5


def test_mutating_holdout_cannot_change_threshold_or_detector(tmp_path):
    demo = load_data(DATASET_PATH, require_electrical=True)
    first = train_project(demo, tmp_path / "first", tmp_path / "first/out", n_estimators=8)
    changed = demo.copy()
    changed.loc[2304:, "active_power_kw"] *= 4
    changed.loc[2304:, "is_anomaly"] = 1 - changed.loc[2304:, "is_anomaly"]
    second = train_project(changed, tmp_path / "second", tmp_path / "second/out", n_estimators=8)
    assert first["anomaly_threshold_calibration"] == second["anomaly_threshold_calibration"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "first/out/anomaly_thresholds.csv"),
                                  pd.read_csv(tmp_path / "second/out/anomaly_thresholds.csv"))
    a, b = EnergyPipeline.load(tmp_path / "first"), EnergyPipeline.load(tmp_path / "second")
    assert a.anomaly["train_end"] == "2026-01-18T23:45:00"
    assert a.anomaly["threshold"] == b.anomaly["threshold"]
    assert a.anomaly["score_min"] == b.anomaly["score_min"]
    assert a.anomaly["score_max"] == b.anomaly["score_max"]
    pd.testing.assert_frame_equal(a.analyze(demo), b.analyze(demo))
