import numpy as np
import pandas as pd
import pytest

from src.data_processing import validate_data
from src.feature_engineering import build_forecast_features, fit_hourly_profile, contextual_features
from src.severity import classify_severity, event_message


def series(n=800):
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="15min"),
                         "active_power_kw": np.arange(n, dtype=float) + 1,
                         "voltage_v": 127., "power_factor": .94})


def test_validation_sorts_without_mutating():
    raw = series(4).iloc[::-1].copy()
    clean = validate_data(raw, require_electrical=True)
    assert clean.timestamp.is_monotonic_increasing
    assert raw.timestamp.iloc[0] > raw.timestamp.iloc[-1]


@pytest.mark.parametrize("column,value", [("active_power_kw", np.nan), ("active_power_kw", np.inf),
    ("active_power_kw", -1), ("voltage_v", 0), ("power_factor", 1.1),
    ("is_anomaly", 2), ("timestamp", "bad"), ("timestamp", "2026-01-01 00:03")])
def test_validation_rejects_bad_values(column, value):
    raw = series(4)
    if column == "timestamp":
        raw[column] = raw[column].astype(str)
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_data(raw, require_electrical=True)


def test_missing_empty_and_duplicate_data():
    with pytest.raises(ValueError, match="voltage_v"):
        validate_data(series(4).drop(columns="voltage_v"), require_electrical=True)
    with pytest.raises(ValueError):
        validate_data(series(0))
    with pytest.raises(ValueError, match="duplic"):
        validate_data(pd.concat([series(4), series(4)]))


def test_lags_and_rollings_use_only_the_past():
    raw = series()
    features = build_forecast_features(raw)
    assert features.loc[96, "lag_1"] == 96
    assert features.loc[96, "lag_4"] == 93
    assert features.loc[96, "lag_96"] == 1
    assert features.loc[672, "lag_672"] == 1
    assert features.loc[96, "rolling_mean_4"] == 94.5
    assert features.loc[96, "rolling_mean_96"] == 48.5
    assert features.loc[96, "rolling_std_4"] == pytest.approx(np.std([93, 94, 95, 96], ddof=1))
    changed = raw.copy()
    changed.loc[96:, "active_power_kw"] = 99999
    pd.testing.assert_frame_equal(features.iloc[:97], build_forecast_features(changed).iloc[:97])


def test_gap_does_not_turn_lag_1_into_thirty_minutes():
    raw = series(120).drop(index=98).reset_index(drop=True)
    result = build_forecast_features(raw)
    assert np.isnan(result.loc[98, "lag_1"])
    assert result.loc[98, "lag_4"] == 96


def test_profile_is_frozen_on_training_only():
    profile = fit_hourly_profile(series(96))
    future = series(96)
    future.active_power_kw = 10000
    result = contextual_features(future, profile)
    assert result.loc[0, "potencia_mediana_hora"] == 2.5
    assert result.loc[0, "desvio_potencia"] == 9997.5


@pytest.mark.parametrize("score,detected,expected", [(100,False,"Normal"), (0,True,"Baixo"),
    (39.9,True,"Baixo"), (40,True,"Moderado"), (60,True,"Alto"), (80,True,"Crítico")])
def test_severity_thresholds(score, detected, expected):
    assert classify_severity(score, detected) == expected


def test_message_describes_downward_deviation_without_inventing_cause():
    text = event_message(2., 10., 85., "Crítico")
    assert "-80.0%" in text
    assert "85.0/100" in text
    assert "curto" not in text and "furto" not in text


def test_csv_duplicate_headers_are_rejected():
    import io
    from src.data_processing import load_data
    content = "timestamp,active_power_kw,active_power_kw\n2026-01-01,1,999\n"
    with pytest.raises(ValueError, match="duplic"):
        load_data(io.StringIO(content))


def test_optional_measurements_and_labels_can_be_partial():
    raw = series(4)
    raw["current_a"] = [10., np.nan, 10., np.nan]
    raw["energy_kwh"] = [1., np.nan, 1., np.nan]
    raw["is_anomaly"] = [1., np.nan, 0., np.nan]
    result = validate_data(raw, require_electrical=True)
    assert result.is_anomaly.isna().sum() == 2
    assert result.current_a.isna().sum() == 2


def test_duplicate_csv_header_after_blank_lines():
    import io
    from src.data_processing import load_data
    with pytest.raises(ValueError, match="duplic"):
        load_data(io.StringIO("\n\n timestamp,active_power_kw,active_power_kw\n2026-01-01,1,2\n"))


def test_extra_csv_fields_cannot_become_an_implicit_index():
    import io
    from src.data_processing import load_data
    bad = "timestamp,active_power_kw,voltage_v,power_factor\nunexpected,2026-02-01 12:00,5,127,.94\n"
    with pytest.raises(ValueError, match="campos"):
        load_data(io.StringIO(bad), require_electrical=True)
