"""Paths and JSON output independent of the current working directory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/raw/energy_consumption_30days.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data/outputs"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
INTERVAL_MINUTES = 15


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
