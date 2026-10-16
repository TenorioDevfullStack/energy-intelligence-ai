"""Causal time-series features and frozen training profiles."""
import pandas as pd

TIME_FEATURES = ["hour", "minute", "weekday", "day", "month"]
LAGS = (1, 4, 96, 672)
WINDOWS = (4, 96)
HISTORY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    f"rolling_{stat}_{window}" for window in WINDOWS for stat in ("mean", "std")]
FORECAST_V2_FEATURES = TIME_FEATURES + HISTORY_FEATURES
ANOMALY_FEATURES = ["voltage_v", "power_factor", "active_power_kw", "desvio_potencia", "razao_potencia"]


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = df["timestamp"].dt
    return pd.DataFrame({"hour": timestamp.hour, "minute": timestamp.minute,
        "weekday": timestamp.dayofweek, "day": timestamp.day, "month": timestamp.month}, index=df.index)


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features at t use observed power strictly before t, including in holdout.

    Reindex on the clock so missing readings do not turn lag_96 into >24 hours.
    Missing history remains NaN; inference explicitly falls back to V1.
    """
    power = df.set_index("timestamp")["active_power_kw"].asfreq("15min")
    historical = pd.DataFrame(index=power.index)
    for lag in LAGS:
        historical[f"lag_{lag}"] = power.shift(lag)
    past = power.shift(1)  # Never include the value we are predicting.
    for window in WINDOWS:
        rolling = past.rolling(window, min_periods=window)
        historical[f"rolling_mean_{window}"] = rolling.mean()
        historical[f"rolling_std_{window}"] = rolling.std()
    aligned = historical.reindex(pd.DatetimeIndex(df.timestamp)).reset_index(drop=True)
    aligned.index = df.index
    return pd.concat([time_features(df), aligned], axis=1)


def fit_hourly_profile(train: pd.DataFrame) -> dict:
    """Learn from training rows only; never recalculate on inference batches."""
    hourly = train.groupby(train.timestamp.dt.hour).active_power_kw.median()
    by_day_type = train.groupby([train.timestamp.dt.dayofweek.ge(5), train.timestamp.dt.hour]).active_power_kw.median()
    return {"hourly": hourly.to_dict(), "day_type": by_day_type.to_dict(),
            "global": float(train.active_power_kw.median())}


def typical_power(df: pd.DataFrame, profile: dict, *, day_type: bool = False) -> pd.Series:
    hours = df.timestamp.dt.hour
    expected = hours.map(profile["hourly"]).fillna(profile["global"])
    if day_type:
        keys = list(zip(df.timestamp.dt.dayofweek.ge(5), hours))
        expected = pd.Series([profile["day_type"].get(key, value) for key, value in zip(keys, expected)], index=df.index)
    return expected.astype(float)


def contextual_features(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    result = df.copy()
    result["potencia_mediana_hora"] = typical_power(df, profile)
    result["desvio_potencia"] = result.active_power_kw - result.potencia_mediana_hora
    # Zero-load installations need a finite denominator, not a future observation.
    result["razao_potencia"] = result.active_power_kw / result.potencia_mediana_hora.clip(lower=1e-6)
    return result
