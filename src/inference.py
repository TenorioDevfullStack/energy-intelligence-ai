"""Persisted models shared by notebooks, API and dashboard; no implicit fit."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .anomaly_detection import detect_anomalies
from .data_processing import validate_data
from .forecasting import CANDIDATES, predict_forecast
from .feature_engineering import ANOMALY_FEATURES, FORECAST_V2_FEATURES, TIME_FEATURES
from .utils import MODEL_DIR
from .config import MAX_EVENT_GAP_MINUTES, MIN_EVENT_POINTS
from .events import build_anomaly_events, summarize_anomaly_events


class ModelArtifactError(ValueError):
    """Stored models are corrupt or incompatible with this pipeline."""


@dataclass
class EnergyPipeline:
    anomaly: dict
    forecast: dict

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "EnergyPipeline":
        folder = Path(model_dir)
        names = ("anomaly_model.joblib", "forecast_model.joblib")
        if not all((folder / name).is_file() for name in names):
            raise FileNotFoundError("Modelos não encontrados. Execute python -m src.train primeiro.")
        try:
            anomaly, forecast = (joblib.load(folder / name) for name in names)
            cls._validate_artifacts(anomaly, forecast)
        except Exception as exc:
            # joblib corruption can raise EOFError, KeyError, pickle errors or
            # missing-class errors. Normalize at this external artifact boundary.
            raise ModelArtifactError(f"Artefatos inválidos ou incompatíveis: {exc}. Execute python -m src.train.") from exc
        return cls(anomaly, forecast)

    @staticmethod
    def _validate_artifacts(anomaly: dict, forecast: dict) -> None:
        required = ((anomaly, {"model", "profile", "features", "score_min", "score_max", "train_end"}),
                    (forecast, {"v1", "v2", "profile", "v2_features", "selected", "train_end"}))
        for bundle, keys in required:
            if not isinstance(bundle, dict) or not keys.issubset(bundle):
                raise ValueError("Estrutura do modelo incompatível")
            profile = bundle["profile"]
            if not isinstance(profile, dict) or not {"hourly", "day_type", "global"}.issubset(profile):
                raise ValueError("Perfil histórico incompatível")
            if not isinstance(profile["hourly"], dict) or not isinstance(profile["day_type"], dict):
                raise ValueError("Perfil histórico incompatível")
            values = [profile["global"], *profile["hourly"].values(), *profile["day_type"].values()]
            if not np.isfinite(values).all() or any(value < 0 for value in values):
                raise ValueError("Perfil histórico inválido")
        if anomaly["features"] != ANOMALY_FEATURES or forecast["v2_features"] != FORECAST_V2_FEATURES:
            raise ValueError("Features incompatíveis")
        if forecast["selected"] not in CANDIDATES:
            raise ValueError("Candidato de previsão desconhecido")
        if not np.isfinite([anomaly["score_min"], anomaly["score_max"]]).all() or anomaly["score_max"] < anomaly["score_min"]:
            raise ValueError("Calibração do score inválida")
        anomaly_end, forecast_end = pd.Timestamp(anomaly["train_end"]), pd.Timestamp(forecast["train_end"])
        if (pd.isna(anomaly_end) or pd.isna(forecast_end) or anomaly_end.tz is not None or forecast_end.tz is not None
                or anomaly_end > forecast_end
                or pd.Timestamp(forecast.get("anomaly_train_end", forecast_end)) != anomaly_end
                or pd.Timestamp(anomaly.get("forecast_train_end", anomaly_end)) != forecast_end):
            raise ValueError("Modelos com períodos de treino incompatíveis")
        threshold = anomaly.get("threshold")
        if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 100):
            raise ValueError("Threshold de anomalia inválido")
        stage2 = (any(key in anomaly for key in ("threshold", "threshold_calibration", "forecast_train_end", "event_policy", "training_data_sha256"))
                  or any(key in forecast for key in ("anomaly_train_end", "training_data_sha256")))
        if stage2:
            if (not {"threshold", "threshold_calibration", "forecast_train_end", "event_policy", "training_data_sha256"}.issubset(anomaly)
                    or not {"anomaly_train_end", "training_data_sha256"}.issubset(forecast)):
                raise ValueError("Metadados da Etapa 2 incompletos")
            calibration = anomaly["threshold_calibration"]
            required_calibration = {"status", "train_end", "validation_start", "validation_end", "holdout_start",
                "selected_threshold", "default_threshold", "threshold_grid", "min_recall", "validation_rows",
                "labelled_rows", "positive_labels", "negative_labels"}
            if not isinstance(calibration, dict) or not required_calibration.issubset(calibration):
                raise ValueError("Metadados da calibração inválidos")
            hashes = [bundle["training_data_sha256"] for bundle in (anomaly, forecast)]
            if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes) or hashes[0] != hashes[1]:
                raise ValueError("Modelos com datasets de treino incompatíveis")
            status = calibration["status"]
            if status not in {"calibrated", "no_labels", "insufficient_classes", "no_feasible_threshold"}:
                raise ValueError("Status de calibração inválido")
            if (status == "calibrated") != (threshold is not None):
                raise ValueError("Status e threshold de calibração incompatíveis")
            grid = np.asarray(calibration["threshold_grid"], dtype=float)
            if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or ((grid < 0) | (grid > 100)).any()
                    or threshold is not None and threshold not in grid
                    or not np.isfinite(calibration["min_recall"]) or not 0 <= calibration["min_recall"] <= 1
                    or not np.isfinite(calibration["default_threshold"]) or not 0 <= calibration["default_threshold"] <= 100):
                raise ValueError("Política de calibração inválida")
            policy = anomaly["event_policy"]
            if not isinstance(policy, dict) or set(policy) != {"max_gap_minutes", "min_event_points"}:
                raise ValueError("Política de eventos inválida")
            build_anomaly_events(pd.DataFrame(), **policy)
            validation_start = pd.Timestamp(calibration["validation_start"])
            validation_end = pd.Timestamp(calibration["validation_end"])
            holdout_start = pd.Timestamp(calibration["holdout_start"])
            if (pd.Timestamp(calibration["train_end"]) != anomaly_end
                    or not anomaly_end < validation_start <= validation_end < holdout_start
                    or validation_end != forecast_end or calibration["selected_threshold"] != threshold
                    or holdout_start != forecast_end + pd.Timedelta(minutes=15)):
                raise ValueError("Calibração temporal ou artefatos incompatíveis")
        for model, expected_type, features in ((anomaly["model"], IsolationForest, ANOMALY_FEATURES),
                (forecast["v1"], RandomForestRegressor, TIME_FEATURES),
                (forecast["v2"], RandomForestRegressor, FORECAST_V2_FEATURES)):
            if not isinstance(model, expected_type):
                raise ValueError("Tipo de modelo incompatível")
            check_is_fitted(model)
            if list(model.feature_names_in_) != features:
                raise ValueError("Modelo ajustado com features incompatíveis")

    def save(self, model_dir: Path = MODEL_DIR) -> None:
        folder = Path(model_dir)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.anomaly, folder / "anomaly_model.joblib", compress=3)
        joblib.dump(self.forecast, folder / "forecast_model.joblib", compress=3)

    def _combine_history(self, rows: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
        if history is None or history.empty:
            return rows.copy()
        previous = validate_data(history)
        if previous.timestamp.max() >= rows.timestamp.min():
            raise ValueError("Todo o histórico deve ser anterior à primeira medição analisada.")
        # Only forecasting columns are needed. Avoid optional-column NaNs after concat.
        columns = ["timestamp", "active_power_kw"]
        return validate_data(pd.concat([previous[columns], rows[columns]], ignore_index=True))

    def analyze(self, df: pd.DataFrame, *, history: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = validate_data(df, require_electrical=True)
        combined = self._combine_history(rows, history)
        predicted = predict_forecast(combined, self.forecast).iloc[-len(rows):].reset_index(drop=True)
        result = detect_anomalies(rows, self.anomaly)
        result["expected_power_kw"] = predicted.expected_power_kw
        result["forecast_source"] = predicted.forecast_source
        result["forecast_error_kw"] = result.active_power_kw - result.expected_power_kw
        result["model_training_overlap"] = result.timestamp <= pd.Timestamp(self.forecast["train_end"])
        result["anomaly_training_overlap"] = result.timestamp <= pd.Timestamp(self.anomaly["train_end"])
        return result

    def analyze_with_events(self, df: pd.DataFrame, *, history: pd.DataFrame | None = None,
                            max_gap_minutes: float | None = None,
                            min_event_points: int | None = None) -> dict:
        """Keep point analysis and add stateless event aggregates for this batch."""
        results = self.analyze(df, history=history)
        policy = self.anomaly.get("event_policy", {})
        max_gap_minutes = policy.get("max_gap_minutes", MAX_EVENT_GAP_MINUTES) if max_gap_minutes is None else max_gap_minutes
        min_event_points = policy.get("min_event_points", MIN_EVENT_POINTS) if min_event_points is None else min_event_points
        events = build_anomaly_events(results, max_gap_minutes=max_gap_minutes, min_event_points=min_event_points)
        return {"results": results, "events": events, "summary": summarize_anomaly_events(events)}

    def forecast_next(self, history: pd.DataFrame, timestamp=None) -> dict:
        """Predict the next quarter-hour using supplied actual readings only."""
        previous = validate_data(history)
        next_time = previous.timestamp.iloc[-1] + pd.Timedelta(minutes=15)
        if timestamp is not None and pd.Timestamp(timestamp) != next_time:
            raise ValueError("A previsão exige o próximo intervalo de 15 minutos após o histórico.")
        row = pd.DataFrame({"timestamp": [next_time], "active_power_kw": [0.]})
        combined = self._combine_history(row, previous)
        # Placeholder target never enters its own features; all history features shift(1).
        predicted = predict_forecast(combined, self.forecast).iloc[-1]
        return {"timestamp": next_time.isoformat(), "predicted_power_kw": float(predicted.expected_power_kw),
                "forecast_source": predicted.forecast_source,
                "model_training_overlap": bool(next_time <= pd.Timestamp(self.forecast["train_end"]))}
