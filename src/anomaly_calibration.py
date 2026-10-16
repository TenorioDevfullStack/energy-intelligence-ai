"""Threshold selection confined to an explicitly bounded temporal validation set."""
import numpy as np
import pandas as pd

from .anomaly_detection import evaluate_anomaly_model
from .config import MIN_RECALL, THRESHOLD_GRID

CALIBRATION_COLUMNS = ("threshold", "evaluated_rows", "precision", "recall", "f1",
                       "false_positive_rate", "specificity", "balanced_accuracy",
                       "false_positives", "false_negatives", "true_positives", "true_negatives")


def calibrate_anomaly_threshold(validation: pd.DataFrame, *, train_end, holdout_start,
                                thresholds=THRESHOLD_GRID, min_recall: float = MIN_RECALL
                                ) -> tuple[float | None, pd.DataFrame, dict]:
    """Maximize validation F1 subject to recall, then fewer FP and higher cut.

    No feasible policy or missing classes/labels returns None: keep model.predict.
    Mandatory temporal boundaries reject training and holdout rows. Scores and
    hourly profiles must already have been frozen on training observations.
    """
    grid = np.asarray(thresholds, dtype=float)
    if not np.isfinite(min_recall) or not 0 <= min_recall <= 1:
        raise ValueError("min_recall deve estar entre 0 e 1.")
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 100)).any():
        raise ValueError("A grade deve conter thresholds finitos entre 0 e 100.")
    if validation.empty or not {"timestamp", "anomaly_score"}.issubset(validation):
        raise ValueError("Envie scores e timestamps do conjunto de validação.")
    times = pd.to_datetime(validation.timestamp, errors="raise", format="mixed")
    start, finish = pd.Timestamp(train_end), pd.Timestamp(holdout_start)
    if times.isna().any() or times.dt.tz is not None or start.tz is not None or finish.tz is not None:
        raise ValueError("Use horários locais válidos na validação.")
    if not start < times.min() <= times.max() < finish:
        raise ValueError("A calibração aceita somente validação posterior ao treino e anterior ao holdout.")
    scores = pd.to_numeric(validation.anomaly_score, errors="raise")
    if not np.isfinite(scores).all() or not scores.between(0, 100).all():
        raise ValueError("Scores da validação devem estar entre 0 e 100.")
    metadata = {"train_end": start.isoformat(), "validation_start": times.min().isoformat(),
                "validation_end": times.max().isoformat(), "holdout_start": finish.isoformat(),
                "validation_rows": len(validation), "min_recall": float(min_recall),
                "threshold_grid": np.unique(grid).tolist(), "labelled_rows": 0,
                "positive_labels": 0, "negative_labels": 0, "selected_threshold": None,
                "policy": "max F1 with recall >= min_recall; ties: fewer FP, higher threshold"}
    empty = pd.DataFrame(columns=CALIBRATION_COLUMNS)
    labelled = validation.is_anomaly.dropna() if "is_anomaly" in validation else pd.Series(dtype=float)
    if not labelled.isin([0, 1]).all():
        raise ValueError("Ground truth de validação aceita apenas 0 e 1.")
    metadata.update(labelled_rows=len(labelled), positive_labels=int(labelled.eq(1).sum()),
                    negative_labels=int(labelled.eq(0).sum()))
    if not len(labelled):
        return None, empty, dict(metadata, status="no_labels")
    if not metadata["positive_labels"] or not metadata["negative_labels"]:
        return None, empty, dict(metadata, status="insufficient_classes")
    rows = []
    for threshold in np.unique(grid):
        metrics = evaluate_anomaly_model(validation, scores.ge(threshold).astype(int))
        rows.append({"threshold": float(threshold), **{name: metrics[name] for name in CALIBRATION_COLUMNS[1:]}})
    table = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)
    feasible = table.loc[table.recall.ge(min_recall)]
    if feasible.empty:
        return None, table, dict(metadata, status="no_feasible_threshold")
    selected = feasible.sort_values(["f1", "false_positives", "threshold"], ascending=[False, True, False]).iloc[0]
    threshold = float(selected.threshold)
    return threshold, table, dict(metadata, status="calibrated", selected_threshold=threshold)
