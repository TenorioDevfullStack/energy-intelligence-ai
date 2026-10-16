"""One-step load forecasts selected using training-only temporal validation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .feature_engineering import (TIME_FEATURES, FORECAST_V2_FEATURES,
    build_forecast_features, fit_hourly_profile, typical_power, time_features)
from .utils import RANDOM_STATE

CANDIDATES = ("forecast_v1", "forecast_v2", "baseline_daily", "baseline_profile")


def evaluate_forecast(truth, prediction) -> dict:
    return {"mae_kw": float(mean_absolute_error(truth, prediction)),
            "rmse_kw": float(np.sqrt(mean_squared_error(truth, prediction))),
            "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else None}


def _fit(train: pd.DataFrame, features: pd.DataFrame, n_estimators: int) -> dict:
    v1 = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    v1.fit(features[TIME_FEATURES], train.active_power_kw)
    available = features[FORECAST_V2_FEATURES].notna().all(axis=1)
    if available.sum() < 32:
        raise ValueError("Histórico insuficiente para Forecast V2: inclua pelo menos 8 dias contínuos.")
    v2 = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=4,
        random_state=RANDOM_STATE, n_jobs=-1)
    v2.fit(features.loc[available, FORECAST_V2_FEATURES], train.loc[available, "active_power_kw"])
    return {"v1": v1, "v2": v2, "profile": fit_hourly_profile(train),
            "v2_features": FORECAST_V2_FEATURES}


def candidate_predictions(features: pd.DataFrame, rows: pd.DataFrame, bundle: dict) -> dict:
    """Call only for rows with complete historical features."""
    return {"forecast_v1": bundle["v1"].predict(features[TIME_FEATURES]),
            "forecast_v2": bundle["v2"].predict(features[bundle["v2_features"]]),
            "baseline_daily": features.lag_96.to_numpy(),
            "baseline_profile": typical_power(rows, bundle["profile"], day_type=True).to_numpy()}


def train_forecast_models(df: pd.DataFrame, *, n_estimators: int = 200) -> tuple[dict, dict, pd.DataFrame]:
    """Final 20% is untouched by model selection; observed past is allowed in test.

    These are rolling one-step predictions, not an open-loop six-day forecast.
    No imputer, normalization or profile sees validation/holdout observations.
    """
    if len(df) < 1200:
        raise ValueError("Treinamento e validação temporal exigem pelo menos 1.200 medições (12,5 dias).")
    features = build_forecast_features(df)
    split = int(len(df) * .8)
    train, test = df.iloc[:split], df.iloc[split:]
    folds = []
    cv_scores = {name: [] for name in CANDIDATES}
    test_size = min(384, (split - 704) // 3)
    for train_idx, validation_idx in TimeSeriesSplit(n_splits=3, test_size=test_size).split(train):
        fold_model = _fit(train.iloc[train_idx], features.iloc[train_idx], n_estimators)
        fold_rows, fold_features = df.iloc[validation_idx], features.iloc[validation_idx]
        predictions = candidate_predictions(fold_features, fold_rows, fold_model)
        fold_metrics = {name: evaluate_forecast(fold_rows.active_power_kw, pred)
                        for name, pred in predictions.items()}
        for name in CANDIDATES:
            cv_scores[name].append(fold_metrics[name])
        folds.append({"train_end": df.timestamp.iloc[train_idx[-1]].isoformat(),
                      "validation_start": df.timestamp.iloc[validation_idx[0]].isoformat(),
                      "validation_end": df.timestamp.iloc[validation_idx[-1]].isoformat(),
                      "train_rows": len(train_idx), "validation_rows": len(validation_idx),
                      "metrics": fold_metrics})
    cv = {name: {metric: {"mean": float(np.mean([x[metric] for x in scores])),
                          "std": float(np.std([x[metric] for x in scores]))}
                  for metric in ("mae_kw", "rmse_kw", "r2")} for name, scores in cv_scores.items()}
    selected = min(CANDIDATES, key=lambda name: cv[name]["mae_kw"]["mean"])
    bundle = _fit(train, features.iloc[:split], n_estimators)
    bundle.update({"selected": selected, "train_end": train.timestamp.iloc[-1].isoformat()})
    predictions = candidate_predictions(features.iloc[split:], test, bundle)
    holdout = {name: evaluate_forecast(test.active_power_kw, pred) for name, pred in predictions.items()}
    results = test[["timestamp", "active_power_kw"]].copy()
    for name, pred in predictions.items():
        results[name] = pred
    results["expected_power_kw"] = results[selected]
    importances = dict(zip(bundle["v2_features"], bundle["v2"].feature_importances_.tolist()))
    report = {"train_rows": split, "holdout_rows": len(test),
        "train_end": train.timestamp.iloc[-1].isoformat(),
        "holdout_start": test.timestamp.iloc[0].isoformat(),
        "holdout_end": test.timestamp.iloc[-1].isoformat(), "forecast_selection": selected,
        "forecast_holdout": holdout, "forecast_cv": cv, "forecast_cv_folds": folds,
        "forecast_v2_features": bundle["v2_features"], "feature_importance_v2": importances,
        "protocol": "One-step ahead; final 20% holdout; selection by training-only 3-fold CV MAE."}
    return bundle, report, results


def predict_forecast(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Use the selected candidate whenever its own features are available.

    V1 and the frozen profile need no supplied history. The daily baseline needs
    only lag_96; V2 needs all its historical features. Missing required history
    uses V1 explicitly, without recalculating any training profile.
    """
    selected = bundle["selected"]
    if selected in ("forecast_v1", "baseline_profile"):
        values = (bundle["v1"].predict(time_features(df)[TIME_FEATURES])
                  if selected == "forecast_v1"
                  else typical_power(df, bundle["profile"], day_type=True).to_numpy())
        return pd.DataFrame({"expected_power_kw": values, "forecast_source": selected}, index=df.index)

    features = build_forecast_features(df)
    prediction = np.asarray(bundle["v1"].predict(features[TIME_FEATURES]), dtype=float)
    source = np.full(len(df), "forecast_v1_fallback", dtype=object)
    required = bundle["v2_features"] if selected == "forecast_v2" else ["lag_96"]
    available = features[required].notna().all(axis=1)
    if available.any():
        good_features = features.loc[available]
        if selected == "forecast_v2":
            values = bundle["v2"].predict(good_features[bundle["v2_features"]])
        else:
            values = good_features.lag_96.to_numpy()
        prediction[available.to_numpy()] = values
        source[available.to_numpy()] = selected
    return pd.DataFrame({"expected_power_kw": prediction, "forecast_source": source}, index=df.index)
