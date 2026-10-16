"""Reproducible training entry point: python -m src.train."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path

import pandas as pd

from .anomaly_detection import train_anomaly_model, detect_anomalies, evaluate_anomaly_model, default_anomaly_threshold
from .anomaly_calibration import calibrate_anomaly_threshold
from .config import ANOMALY_TRAIN_FRACTION, MIN_RECALL, THRESHOLD_GRID, MAX_EVENT_GAP_MINUTES, MIN_EVENT_POINTS
from .data_processing import load_data, validate_data
from .forecasting import train_forecast_models
from .inference import EnergyPipeline
from .events import build_anomaly_events, summarize_anomaly_events
from .utils import DATASET_PATH, MODEL_DIR, OUTPUT_DIR, REPORT_DIR, write_json


def train_project(df: pd.DataFrame, model_dir: Path = MODEL_DIR, output_dir: Path = OUTPUT_DIR,
                  *, n_estimators: int = 200, min_recall: float = MIN_RECALL,
                  thresholds=THRESHOLD_GRID, min_event_points: int = MIN_EVENT_POINTS,
                  max_event_gap_minutes: float = MAX_EVENT_GAP_MINUTES) -> dict:
    rows = validate_data(df, require_electrical=True, require_regular=True)
    forecast, report, forecast_results = train_forecast_models(rows, n_estimators=n_estimators)
    split = report["train_rows"]
    anomaly_split = int(len(rows) * ANOMALY_TRAIN_FRACTION)
    train, validation, final_test = rows.iloc[:anomaly_split], rows.iloc[anomaly_split:split], rows.iloc[split:]
    anomaly = train_anomaly_model(train)
    # Only validation scores enter selection. No refit follows calibration: the
    # frozen profile and score scale remain exactly those used by the threshold.
    validation_default = detect_anomalies(validation, anomaly)
    threshold, threshold_table, calibration = calibrate_anomaly_threshold(
        validation_default, train_end=anomaly["train_end"], holdout_start=final_test.timestamp.iloc[0],
        thresholds=thresholds, min_recall=min_recall)
    calibration["default_threshold"] = default_anomaly_threshold(anomaly)
    anomaly.update(threshold=threshold, threshold_calibration=calibration,
                   forecast_train_end=forecast["train_end"])
    forecast["anomaly_train_end"] = anomaly["train_end"]
    data_hash = hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()
    anomaly["training_data_sha256"] = forecast["training_data_sha256"] = data_hash
    pipeline = EnergyPipeline(anomaly, forecast)
    results = pipeline.analyze(rows)
    holdout = results.iloc[split:]
    report["anomaly_holdout"] = evaluate_anomaly_model(holdout, holdout.is_anomaly_prediction)
    report["anomaly_train_diagnostic"] = evaluate_anomaly_model(results.iloc[:anomaly_split], results.is_anomaly_prediction.iloc[:anomaly_split])
    validation_calibrated = detect_anomalies(validation, anomaly)
    report["anomaly_comparison"] = {
        "validation": {"default": evaluate_anomaly_model(validation_default, validation_default.is_anomaly_prediction),
                       "calibrated": evaluate_anomaly_model(validation_calibrated, validation_calibrated.is_anomaly_prediction)},
        "holdout": {"default": evaluate_anomaly_model(holdout, holdout.is_anomaly_prediction_default),
                    "calibrated": report["anomaly_holdout"]}}
    # Reproduce the previous 80%-fit V2 as a historical reference only. Its
    # holdout results never enter calibration or the persisted calibrated model.
    legacy = train_anomaly_model(rows.iloc[:split])
    legacy_holdout = detect_anomalies(final_test, legacy)
    report["anomaly_legacy"] = {"train_rows": split, "train_end": legacy["train_end"],
        "default_threshold": default_anomaly_threshold(legacy),
        "holdout": evaluate_anomaly_model(legacy_holdout, legacy_holdout.is_anomaly_prediction)}
    report["anomaly_threshold_calibration"] = calibration
    report["anomaly_split"] = {"train_rows": len(train), "validation_rows": len(validation), "holdout_rows": len(final_test),
        "train_start": train.timestamp.iloc[0].isoformat(), "train_end": anomaly["train_end"],
        "validation_start": validation.timestamp.iloc[0].isoformat(), "validation_end": validation.timestamp.iloc[-1].isoformat(),
        "holdout_start": final_test.timestamp.iloc[0].isoformat(), "holdout_end": final_test.timestamp.iloc[-1].isoformat()}
    report["anomaly_contamination"] = anomaly["contamination"]
    report["score_calibration"] = {"min": anomaly["score_min"], "max": anomaly["score_max"],
        "method": "100 * (negative decision_function - train_min) / (train_max - train_min), clipped [0,100]"}
    report["versions"] = {name: importlib.metadata.version(name)
                           for name in ("numpy", "pandas", "scikit-learn", "joblib")}
    report["training_data_sha256"] = data_hash
    report["n_estimators"] = n_estimators
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    event_options = {"max_gap_minutes": max_event_gap_minutes, "min_event_points": min_event_points}
    anomaly["event_policy"] = event_options
    events = build_anomaly_events(results, **event_options)
    report["operational_events"] = {"all": summarize_anomaly_events(events),
        "holdout": summarize_anomaly_events(build_anomaly_events(holdout, **event_options)),
        "policy": event_options}
    pipeline.save(model_dir)
    results.to_csv(output / "analysis_results.csv", index=False)
    results.loc[results.is_anomaly_prediction.eq(1)].to_csv(output / "anomaly_points.csv", index=False)
    events.to_csv(output / "anomaly_events.csv", index=False)
    threshold_table.to_csv(output / "anomaly_thresholds.csv", index=False)
    forecast_results.to_csv(output / "forecast_results.csv", index=False)
    write_json(Path(model_dir) / "metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Energy Intelligence AI MVP")
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--outputs", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    parser.add_argument("--min-event-points", type=int, default=MIN_EVENT_POINTS)
    parser.add_argument("--max-event-gap-minutes", type=float, default=MAX_EVENT_GAP_MINUTES)
    args = parser.parse_args()
    report = train_project(load_data(args.data, require_electrical=True, require_regular=True), args.models, args.outputs,
        min_recall=args.min_recall, min_event_points=args.min_event_points, max_event_gap_minutes=args.max_event_gap_minutes)
    write_json(REPORT_DIR / "model_metrics.json", report)
    table = pd.DataFrame(report["forecast_holdout"]).T
    print("Forecast holdout (final 20%):")
    print(table.round(4).to_string())
    print("Selected by training-only CV:", report["forecast_selection"])
    print("Anomaly holdout:", report["anomaly_holdout"])
    print("Anomaly threshold:", report["anomaly_threshold_calibration"]["selected_threshold"])
    print("Anomaly validation:", report["anomaly_comparison"]["validation"])
    print("Anomaly holdout before/after (same fitted detector):", report["anomaly_comparison"]["holdout"])
    print("Events on holdout:", report["operational_events"]["holdout"])
    print("Models:", args.models)
    print("Outputs:", args.outputs)


if __name__ == "__main__":
    main()
