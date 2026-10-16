"""Isolation Forest V2 with frozen hourly context and score calibration."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from .feature_engineering import ANOMALY_FEATURES, contextual_features, fit_hourly_profile
from .severity import classify_severity, event_message
from .utils import RANDOM_STATE

# Demo's known insertion rate is a configuration, never inferred from labels.
DEFAULT_CONTAMINATION = 20 / 2880


def train_anomaly_model(df: pd.DataFrame, *, contamination: float = DEFAULT_CONTAMINATION) -> dict:
    profile = fit_hourly_profile(df)
    features = contextual_features(df, profile)[ANOMALY_FEATURES]
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(features)
    raw = -model.decision_function(features)
    return {"model": model, "profile": profile, "features": ANOMALY_FEATURES,
            "score_min": float(raw.min()), "score_max": float(raw.max()),
            "train_end": df.timestamp.max().isoformat(), "contamination": contamination}


def detect_anomalies(df: pd.DataFrame, bundle: dict, *, use_calibrated_threshold: bool = True) -> pd.DataFrame:
    result = contextual_features(df, bundle["profile"])
    features = result[bundle["features"]]
    default_prediction = (bundle["model"].predict(features) == -1).astype(int)
    raw = -bundle["model"].decision_function(features)
    span = max(bundle["score_max"] - bundle["score_min"], 1e-12)
    # Calibration belongs to training, not each request. Clip new extremes.
    result["anomaly_score"] = np.clip(100 * (raw - bundle["score_min"]) / span, 0, 100)
    threshold = bundle.get("threshold") if use_calibrated_threshold else None
    result["is_anomaly_prediction_default"] = default_prediction
    result["is_anomaly_prediction"] = (result.anomaly_score.ge(threshold).astype(int)
                                        if threshold is not None else default_prediction)
    result["severity"] = [classify_severity(float(score), bool(flag))
        for score, flag in zip(result.anomaly_score, result.is_anomaly_prediction)]
    result["event_message"] = [event_message(float(power), float(typical), float(score), severity)
        for power, typical, score, severity in zip(result.active_power_kw,
            result.potencia_mediana_hora, result.anomaly_score, result.severity)]
    return result


def evaluate_anomaly_model(df: pd.DataFrame, prediction) -> dict | None:
    """Supervised evaluation is available only with explicit ground truth."""
    if "is_anomaly" not in df:
        return None
    labelled = df.is_anomaly.notna()
    if not labelled.any():
        return None
    truth = df.loc[labelled, "is_anomaly"].astype(int)
    prediction = np.asarray(prediction)[labelled.to_numpy()]
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else None
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    return {"evaluated_rows": int(labelled.sum()),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "true_positives": int(tp), "false_positives": int(fp),
            "false_negatives": int(fn), "true_negatives": int(tn),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "specificity": specificity,
            "balanced_accuracy": (specificity + sensitivity) / 2 if specificity is not None and sensitivity is not None else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def default_anomaly_threshold(bundle: dict) -> float:
    """Default decision_function < 0 expressed on this model's frozen score scale."""
    span = max(bundle["score_max"] - bundle["score_min"], 1e-12)
    return float(np.clip(-100 * bundle["score_min"] / span, 0, 100))
