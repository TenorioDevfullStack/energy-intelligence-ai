"""FastAPI inference service. Start with uvicorn app.api:app --reload."""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.anomaly_detection import detect_anomalies, evaluate_anomaly_model
from src.data_processing import validate_data
from src.inference import EnergyPipeline
from src.utils import MODEL_DIR


class HistoricalReading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: datetime
    active_power_kw: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def local_quarter_hour(cls, value: datetime) -> datetime:
        if value.tzinfo is not None or value.minute % 15 or value.second or value.microsecond:
            raise ValueError("Use horário local sem timezone, alinhado a 15 minutos.")
        return value


class ElectricalReading(HistoricalReading):
    voltage_v: float = Field(gt=0)
    power_factor: float = Field(gt=0, le=1)
    current_a: float | None = Field(default=None, ge=0)
    energy_kwh: float | None = Field(default=None, ge=0)
    is_anomaly: Literal[0, 1] | None = None


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[ElectricalReading] = Field(min_length=1, max_length=5000)
    history: list[HistoricalReading] = Field(default_factory=list, max_length=4096)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: list[HistoricalReading] = Field(min_length=1, max_length=4096)
    timestamp: datetime | None = None


class EventAnalysisRequest(AnalysisRequest):
    max_gap_minutes: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    min_event_points: int | None = Field(default=None, ge=1, strict=True)


class OperationalEvent(BaseModel):
    event_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    duration_minutes: float = Field(ge=0)
    number_of_points: int = Field(ge=1)
    max_active_power_kw: float = Field(ge=0)
    mean_active_power_kw: float = Field(ge=0)
    expected_power_kw: float | None
    max_relative_deviation_pct: float | None
    max_anomaly_score: float = Field(ge=0, le=100)
    mean_anomaly_score: float = Field(ge=0, le=100)
    severity: Literal["Baixo", "Moderado", "Alto", "Crítico"]
    status: Literal["suspected", "confirmed"]
    message: str


class EventSummary(BaseModel):
    total_events: int
    confirmed_events: int
    suspected_events: int
    critical_events: int
    total_duration_minutes: float
    severity_distribution: dict[str, int]


class EventAnalysisResponse(BaseModel):
    summary: EventSummary
    events: list[OperationalEvent]
    anomaly_metrics: dict | None


class AnomalyResult(BaseModel):
    timestamp: datetime
    active_power_kw: float
    potencia_mediana_hora: float
    is_anomaly_prediction: int
    anomaly: bool
    anomaly_score: float = Field(ge=0, le=100)
    severity: Literal["Normal", "Baixo", "Moderado", "Alto", "Crítico"]
    event_message: str
    message: str
    model_training_overlap: bool


class AnalysisResult(AnomalyResult):
    expected_power_kw: float
    predicted_power_kw: float
    forecast_error_kw: float
    forecast_source: str


class AnalysisResponse(BaseModel):
    results: list[AnalysisResult]
    anomaly_metrics: dict | None


class AnomalyResponse(BaseModel):
    results: list[AnomalyResult]
    anomaly_metrics: dict | None


class ForecastResponse(BaseModel):
    timestamp: datetime
    predicted_power_kw: float
    forecast_source: str
    model_training_overlap: bool


def _frame(readings: list[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump(exclude_none=True) for row in readings])


def _pipeline(request: Request) -> EnergyPipeline:
    if request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail=request.app.state.model_error)
    return request.app.state.pipeline


def _records(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict(orient="records"):
        row["timestamp"] = row["timestamp"].isoformat()
        row["anomaly"] = bool(row["is_anomaly_prediction"])
        row["message"] = row["event_message"]
        if "expected_power_kw" in row:
            row["predicted_power_kw"] = row["expected_power_kw"]
        rows.append(row)
    return rows


def create_app(model_dir: Path = MODEL_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(instance: FastAPI):
        try:
            instance.state.pipeline = EnergyPipeline.load(model_dir)
            instance.state.model_error = None
        except (FileNotFoundError, OSError, ValueError, EOFError, ImportError) as exc:
            instance.state.pipeline = None
            instance.state.model_error = f"Modelos indisponíveis: {exc}"
        yield

    instance = FastAPI(title="Energy Intelligence AI", version="0.3.0", lifespan=lifespan,
        description="Detecção contextual de anomalias e previsão de potência de um passo (15 min).")

    @instance.get("/health")
    def health(request: Request):
        ready = request.app.state.pipeline is not None
        payload = {"status": "ok" if ready else "unavailable", "models_ready": ready}
        if ready:
            payload["forecast_model"] = request.app.state.pipeline.forecast["selected"]
            payload["trained_until"] = request.app.state.pipeline.forecast["train_end"]
            detector = request.app.state.pipeline.anomaly
            payload["anomaly_trained_until"] = detector["train_end"]
            payload["anomaly_threshold"] = detector.get("threshold")
            payload["anomaly_calibration_status"] = detector.get("threshold_calibration", {}).get("status", "legacy_default")
        else:
            payload["detail"] = request.app.state.model_error
        return JSONResponse(payload, status_code=200 if ready else 503)

    @instance.post("/analyze", response_model=AnalysisResponse)
    def analyze(payload: AnalysisRequest, pipeline: EnergyPipeline = Depends(_pipeline)):
        try:
            result = pipeline.analyze(_frame(payload.records), history=_frame(payload.history))
            return {"results": _records(result),
                    "anomaly_metrics": evaluate_anomaly_model(result, result.is_anomaly_prediction)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @instance.post("/predict/anomaly", response_model=AnomalyResponse)
    def anomaly(payload: AnalysisRequest, pipeline: EnergyPipeline = Depends(_pipeline)):
        try:
            data = validate_data(_frame(payload.records), require_electrical=True)
            result = detect_anomalies(data, pipeline.anomaly)
            calibration_end = pipeline.anomaly.get("threshold_calibration", {}).get("validation_end", pipeline.anomaly["train_end"])
            result["model_training_overlap"] = result.timestamp <= pd.Timestamp(calibration_end)
            return {"results": _records(result),
                    "anomaly_metrics": evaluate_anomaly_model(result, result.is_anomaly_prediction)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @instance.post("/analyze/events", response_model=EventAnalysisResponse)
    def analyze_events(payload: EventAnalysisRequest, pipeline: EnergyPipeline = Depends(_pipeline)):
        try:
            analysis = pipeline.analyze_with_events(_frame(payload.records), history=_frame(payload.history),
                max_gap_minutes=payload.max_gap_minutes, min_event_points=payload.min_event_points)
            # Nullable references must serialize as null, never NaN.
            events = analysis["events"].astype(object).where(pd.notna(analysis["events"]), None).to_dict(orient="records")
            return {"summary": analysis["summary"], "events": events,
                "anomaly_metrics": evaluate_anomaly_model(analysis["results"], analysis["results"].is_anomaly_prediction)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @instance.post("/predict/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest, pipeline: EnergyPipeline = Depends(_pipeline)):
        try:
            return pipeline.forecast_next(_frame(payload.history), payload.timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return instance


app = create_app()
