"""Validate the persisted report consumed by every dashboard section."""
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from src.forecasting import CANDIDATES


class ReportModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, strict=True)


# JSON reports use ISO strings for timestamps, while numeric fields remain strict.
ReportTimestamp = Annotated[datetime, Field(strict=False)]


class ForecastScore(ReportModel):
    mae_kw: float = Field(ge=0)
    rmse_kw: float = Field(ge=0)
    r2: float | None


class MeanStd(ReportModel):
    mean: float
    std: float = Field(ge=0)


class ForecastCV(ReportModel):
    mae_kw: MeanStd
    rmse_kw: MeanStd
    r2: MeanStd


class DetectionScore(ReportModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_positive_rate: float | None
    specificity: float | None
    balanced_accuracy: float | None


class ThresholdComparison(ReportModel):
    default: DetectionScore | None
    calibrated: DetectionScore | None


class AnomalyComparison(ReportModel):
    validation: ThresholdComparison
    holdout: ThresholdComparison


class Calibration(ReportModel):
    default_threshold: float = Field(ge=0, le=100)
    selected_threshold: float | None = Field(ge=0, le=100)
    min_recall: float = Field(ge=0, le=1)
    status: str


class AnomalySplit(ReportModel):
    train_start: ReportTimestamp
    train_end: ReportTimestamp
    validation_start: ReportTimestamp
    validation_end: ReportTimestamp
    holdout_start: ReportTimestamp
    holdout_end: ReportTimestamp


class LegacyReport(ReportModel):
    holdout: DetectionScore | None


class DashboardReport(ReportModel):
    train_end: ReportTimestamp
    holdout_start: ReportTimestamp
    holdout_end: ReportTimestamp
    holdout_rows: int = Field(gt=0)
    forecast_selection: str
    forecast_holdout: dict[str, ForecastScore]
    forecast_cv: dict[str, ForecastCV]
    feature_importance_v2: dict[str, float]
    anomaly_holdout: DetectionScore | None
    anomaly_comparison: AnomalyComparison | None = None
    anomaly_threshold_calibration: Calibration | None = None
    anomaly_split: AnomalySplit | None = None
    anomaly_legacy: LegacyReport | None = None


def validate_dashboard_report(metrics: dict, pipeline) -> None:
    report = DashboardReport.model_validate(metrics)
    if (metrics["train_end"] != pipeline.forecast["train_end"]
            or metrics.get("training_data_sha256") != pipeline.forecast.get("training_data_sha256")
            or report.forecast_selection != pipeline.forecast["selected"]):
        raise ValueError("Métricas incompatíveis com os modelos.")
    if (set(report.forecast_holdout) != set(CANDIDATES)
            or set(report.forecast_cv) != set(CANDIDATES)):
        raise ValueError("Relatório de candidatos incompleto.")
    if report.anomaly_comparison is not None:
        calibration, split = report.anomaly_threshold_calibration, report.anomaly_split
        if (calibration is None or split is None
                or calibration.selected_threshold != pipeline.anomaly.get("threshold")
                or split.train_end.isoformat() != pipeline.anomaly["train_end"]
                or split.validation_end != report.train_end
                or not split.train_start <= split.train_end < split.validation_start <= split.validation_end < split.holdout_start <= split.holdout_end):
            raise ValueError("Relatório de calibração incompatível.")
        for comparison in (report.anomaly_comparison.validation, report.anomaly_comparison.holdout):
            if (comparison.default is None) != (comparison.calibrated is None):
                raise ValueError("Comparação de thresholds incompleta.")
