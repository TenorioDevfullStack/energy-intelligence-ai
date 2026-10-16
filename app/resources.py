"""Cached dashboard startup; preparation is separate from inference and uploads."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from src.data_processing import load_data
from src.inference import EnergyPipeline, ModelArtifactError
from app.metrics import validate_dashboard_report

DEMO_SHA256 = "fda66ff898f1c8e348fa42cd1b6c1f29a4091f032258f1b3663341e182df0995"
ASSET_NAMES = ("anomaly_model.joblib", "forecast_model.joblib", "metrics.json")


@dataclass
class DashboardResources:
    pipeline: EnergyPipeline
    metrics: dict
    prepared_demo: bool = False
    preparation_seconds: float = 0.


def resource_signature(model_dir: Path) -> tuple:
    """Include metrics and missing files so replacing artifacts invalidates caches."""
    signature = []
    for name in ASSET_NAMES:
        try:
            stat = (model_dir / name).stat()
            signature.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append(None)
    return tuple(signature)


@st.cache_resource(show_spinner="Preparando a demonstração energética…")
def load_demo_resources(demo_content: bytes) -> DashboardResources:
    """Build the bounded bundled demo once per process, never an uploaded CSV.

    The digest limits automatic training to the original 2,880 synthetic points.
    Temporary artifacts are loaded and removed; configured local models are intact.
    Streamlit serializes concurrent cache misses for this resource.
    """
    if hashlib.sha256(demo_content).hexdigest() != DEMO_SHA256:
        raise ValueError("O preparo automático aceita somente o dataset demo original.")
    from src.train import train_project

    started = perf_counter()
    rows = load_data(io.BytesIO(demo_content), require_electrical=True, require_regular=True)
    with TemporaryDirectory(prefix="energy-demo-") as temporary:
        folder = Path(temporary)
        metrics = train_project(rows, folder, folder / "outputs")
        pipeline = EnergyPipeline.load(folder)
    return DashboardResources(pipeline, metrics, True, perf_counter() - started)


@st.cache_resource(show_spinner=False, max_entries=4)
def load_dashboard_resources(model_dir: str, revision: tuple, demo_content: bytes) -> DashboardResources:
    """Use valid persisted resources, otherwise an explicitly identified demo."""
    folder = Path(model_dir)
    try:
        pipeline = EnergyPipeline.load(folder)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        validate_dashboard_report(metrics, pipeline)
        return DashboardResources(pipeline, metrics)
    except (OSError, ModelArtifactError, ValueError, KeyError, TypeError):
        return load_demo_resources(demo_content)
