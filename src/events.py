"""Batch-scoped operational events derived from point detections, without ML fit."""
import math
from numbers import Integral

import numpy as np
import pandas as pd

from .config import MAX_EVENT_GAP_MINUTES, MIN_EVENT_POINTS
from .data_processing import validate_data
from .severity import SEVERITY_ORDER, aggregate_event_severity

EVENT_COLUMNS = ("event_id", "start_timestamp", "end_timestamp", "duration_minutes",
                 "number_of_points", "max_active_power_kw", "mean_active_power_kw",
                 "expected_power_kw", "max_relative_deviation_pct", "max_anomaly_score",
                 "mean_anomaly_score", "severity", "status", "message")


def operational_event_message(event: dict) -> str:
    """Describe duration, observed load and score; persistence does not imply cause."""
    expected = f"{event['expected_power_kw']:.2f} kW" if event["expected_power_kw"] is not None else "indisponível"
    deviation = (f"{event['max_relative_deviation_pct']:+.1f}%"
                 if event["max_relative_deviation_pct"] is not None else "indisponível (referência próxima de zero ou ausente)")
    return (f"Evento energético anômalo — severidade {event['severity']}. "
            f"Início: {event['start_timestamp']:%d/%m/%Y %H:%M}; fim: {event['end_timestamp']:%d/%m/%Y %H:%M}; "
            f"duração: {event['duration_minutes']:g} min; {event['number_of_points']} pontos. "
            f"Potência máxima: {event['max_active_power_kw']:.2f} kW; potência esperada média: {expected}; "
            f"desvio relativo de maior magnitude: {deviation}; score máximo: {event['max_anomaly_score']:.1f}/100.")


def build_anomaly_events(analysis: pd.DataFrame, *, max_gap_minutes: float = MAX_EVENT_GAP_MINUTES,
                         min_event_points: int = MIN_EVENT_POINTS) -> pd.DataFrame:
    """Group adjacent detections; observed normal points always break an event.

    Duration is elapsed time between first/last readings (singletons: 0 minutes).
    Required persistence changes status only; isolated points are never removed.
    The strongest relative deviation keeps its sign; zero references are omitted.
    """
    if not math.isfinite(max_gap_minutes) or max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes deve ser positivo e finito.")
    try:
        max_gap = pd.Timedelta(minutes=max_gap_minutes)
    except (OverflowError, ValueError) as exc:
        raise ValueError("max_gap_minutes excede o intervalo temporal suportado.") from exc
    if not isinstance(min_event_points, Integral) or isinstance(min_event_points, bool) or min_event_points < 1:
        raise ValueError("min_event_points deve ser um inteiro positivo.")
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    if analysis.empty:
        return empty
    required = {"timestamp", "active_power_kw", "is_anomaly_prediction", "anomaly_score", "severity"}
    if not required.issubset(analysis):
        raise ValueError("Análise sem as colunas necessárias para eventos.")
    rows = validate_data(analysis)
    if not rows.is_anomaly_prediction.isin([0, 1]).all() or not rows.anomaly_score.between(0, 100).all():
        raise ValueError("Detecções ou scores inválidos para eventos.")
    detected = rows.is_anomaly_prediction.eq(1)
    starts = detected & (~detected.shift(fill_value=False) |
                        rows.timestamp.diff().gt(max_gap))
    groups = starts.cumsum()
    events = []
    for _, points in rows.loc[detected].groupby(groups.loc[detected], sort=True):
        start, end = points.timestamp.iloc[0], points.timestamp.iloc[-1]
        reference_column = next((name for name in ("expected_power_kw", "potencia_mediana_hora") if name in points), None)
        expected, deviation = None, None
        if reference_column is not None:
            reference = points[reference_column].astype(float)
            if not np.isfinite(reference).all() or (reference < 0).any():
                raise ValueError("Potência esperada inválida para eventos.")
            expected = float(reference.mean())
            valid = reference.gt(1e-6)
            relative = (points.loc[valid, "active_power_kw"] / reference.loc[valid] - 1) * 100
            if not relative.empty:
                deviation = float(relative.loc[relative.abs().idxmax()])
        event = {"event_id": f"EVT-{start:%Y%m%dT%H%M%S}", "start_timestamp": start,
                 "end_timestamp": end, "duration_minutes": (end - start).total_seconds() / 60,
                 "number_of_points": len(points), "max_active_power_kw": float(points.active_power_kw.max()),
                 "mean_active_power_kw": float(points.active_power_kw.mean()), "expected_power_kw": expected,
                 "max_relative_deviation_pct": deviation, "max_anomaly_score": float(points.anomaly_score.max()),
                 "mean_anomaly_score": float(points.anomaly_score.mean()),
                 "severity": aggregate_event_severity(points.severity),
                 "status": "confirmed" if len(points) >= min_event_points else "suspected"}
        event["message"] = operational_event_message(event)
        events.append(event)
    return pd.DataFrame(events, columns=EVENT_COLUMNS) if events else empty


def summarize_anomaly_events(events: pd.DataFrame) -> dict:
    return {"total_events": len(events), "confirmed_events": int(events.status.eq("confirmed").sum()),
            "suspected_events": int(events.status.eq("suspected").sum()),
            "critical_events": int(events.severity.eq("Crítico").sum()),
            "total_duration_minutes": float(events.duration_minutes.sum()),
            "severity_distribution": {name: int(events.severity.eq(name).sum()) for name in SEVERITY_ORDER}}
