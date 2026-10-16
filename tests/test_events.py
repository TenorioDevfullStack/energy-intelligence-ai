import pandas as pd
import pytest


def readings(flags=(1, 1, 1), severities=("Alto", "Crítico", "Moderado")):
    return pd.DataFrame({"timestamp": pd.date_range("2026-02-01T18:15", periods=len(flags), freq="15min"),
                         "active_power_kw": 18., "expected_power_kw": 9.,
                         "is_anomaly_prediction": flags, "anomaly_score": 90., "severity": severities})


def build(rows, **kwargs):
    from src.events import build_anomaly_events
    return build_anomaly_events(rows, **kwargs)


def test_consecutive_points_become_one_persistent_event():
    from src.events import summarize_anomaly_events
    events = build(readings())
    assert len(events) == 1
    event = events.iloc[0]
    assert event.number_of_points == 3 and event.duration_minutes == 30
    assert event.status == "confirmed" and event.severity == "Crítico"
    assert event.expected_power_kw == 9 and event.max_relative_deviation_pct == 100
    assert event.max_active_power_kw == 18 and event.mean_anomaly_score == 90
    assert "30 min" in event.message and "18.00 kW" in event.message
    assert all(word not in event.message.lower() for word in ("curto", "furto", "sobrecarga", "defeituoso"))
    summary = summarize_anomaly_events(events)
    assert summary["total_events"] == summary["confirmed_events"] == summary["critical_events"] == 1
    assert summary["total_duration_minutes"] == 30


def test_isolated_point_is_retained_as_suspected():
    event = build(readings((1,), ("Alto",))).iloc[0]
    assert event.status == "suspected" and event.duration_minutes == 0
    assert build(readings((1,), ("Alto",)), min_event_points=1).iloc[0].status == "confirmed"


def test_normal_point_breaks_events_even_with_larger_gap_tolerance():
    events = build(readings((1, 0, 1), ("Alto", "Normal", "Crítico")), max_gap_minutes=30)
    assert len(events) == 2 and events.number_of_points.tolist() == [1, 1]


def test_missing_reading_and_configurable_tolerance():
    rows = readings().drop(index=1)
    assert len(build(rows)) == 2
    event = build(rows, max_gap_minutes=30).iloc[0]
    assert event.number_of_points == 2 and event.duration_minutes == 30


def test_empty_events_have_stable_columns_and_zero_summary():
    from src.events import EVENT_COLUMNS, summarize_anomaly_events
    events = build(readings((0, 0, 0), ("Normal",) * 3))
    assert events.empty and list(events.columns) == list(EVENT_COLUMNS)
    assert summarize_anomaly_events(events)["total_events"] == 0
    assert summarize_anomaly_events(events)["total_duration_minutes"] == 0


def test_zero_reference_and_downward_deviation_are_interpretable():
    rows = readings((1,), ("Crítico",))
    rows["expected_power_kw"] = 0.
    event = build(rows).iloc[0]
    assert pd.isna(event.max_relative_deviation_pct)
    assert "indisponível" in event.message
    rows["expected_power_kw"] = 90.
    event = build(rows).iloc[0]
    assert event.max_relative_deviation_pct == -80
    assert "-80.0%" in event.message


def test_event_ids_are_stable_and_input_is_not_mutated():
    rows = readings()
    before = rows.copy()
    event = build(rows).iloc[0]
    assert event.event_id == build(rows.iloc[::-1]).iloc[0].event_id
    pd.testing.assert_frame_equal(rows, before)


def test_mixed_deviations_use_each_points_reference_and_preserve_sign():
    rows = readings()
    rows["expected_power_kw"] = [9., 90., 18.]
    rows["active_power_kw"] = [12., 9., 18.]
    event = build(rows).iloc[0]
    assert event.expected_power_kw == 39.
    assert event.max_relative_deviation_pct == -90.


@pytest.mark.parametrize("kwargs", [{"min_event_points": 0}, {"min_event_points": 1.5},
                                    {"max_gap_minutes": 0}, {"max_gap_minutes": 1e308}])
def test_invalid_event_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build(readings(), **kwargs)
