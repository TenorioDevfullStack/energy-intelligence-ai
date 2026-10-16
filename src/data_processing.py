"""Validate one installation's electrical readings before modeling."""
import csv
import io
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

BASE_COLUMNS = ("timestamp", "active_power_kw")
ELECTRICAL_COLUMNS = ("voltage_v", "power_factor")
MAX_ROWS = 100_000
MAX_SPAN_DAYS = 366


def validate_data(df: pd.DataFrame, *, require_electrical: bool = False,
                  require_regular: bool = False) -> pd.DataFrame:
    """Return a sorted copy; gaps are allowed except during model training.

    Timestamps are local installation time (no timezone); lags assume 15 minutes.
    Negative power is outside this consumption-only MVP's supported contract.
    """
    required = BASE_COLUMNS + (ELECTRICAL_COLUMNS if require_electrical else ())
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(missing))
    if df.empty or len(df) > MAX_ROWS:
        raise ValueError(f"Envie de 1 a {MAX_ROWS} medições.")
    if not df.columns.is_unique:
        raise ValueError("Há colunas duplicadas no arquivo.")
    clean = df.copy()
    try:
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise", format="mixed")
        if clean.timestamp.isna().any() or clean.timestamp.dt.tz is not None:
            raise ValueError("Use horários locais sem timezone e sem valores vazios.")
        if (clean.timestamp != clean.timestamp.dt.floor("15min")).any():
            raise ValueError("timestamp deve estar alinhado a intervalos de 15 minutos.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"timestamp inválido: {exc}") from exc
    if clean.timestamp.duplicated().any():
        raise ValueError("Há timestamps duplicados; envie apenas uma instalação por vez.")
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    if clean.timestamp.iloc[-1] - clean.timestamp.iloc[0] > pd.Timedelta(days=MAX_SPAN_DAYS):
        raise ValueError(f"O intervalo do dataset deve ser de até {MAX_SPAN_DAYS} dias.")
    for column in ("active_power_kw", "voltage_v", "power_factor", "current_a", "energy_kwh", "is_anomaly"):
        if column not in clean:
            continue
        try:
            clean[column] = pd.to_numeric(clean[column], errors="raise")
            # Missing optional measurements/labels are permitted; model inputs are not.
            values = clean[column].dropna() if column in ("current_a", "energy_kwh", "is_anomaly") else clean[column]
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError("valores vazios ou não finitos")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Coluna {column} inválida: {exc}") from exc
        if column == "is_anomaly":
            if not clean[column].dropna().isin([0, 1]).all():
                raise ValueError("is_anomaly aceita apenas 0 e 1.")
            clean[column] = clean[column].astype("Int64")
        elif column == "power_factor":
            if not clean[column].between(0, 1, inclusive="right").all():
                raise ValueError("power_factor deve ser maior que 0 e até 1.")
        elif column == "voltage_v":
            if (clean[column] <= 0).any():
                raise ValueError("voltage_v deve ser maior que zero.")
        elif (clean[column] < 0).any():
            raise ValueError(f"{column} não pode ser negativo.")
    if require_regular and len(clean) > 1:
        if not clean.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
            raise ValueError("Treinamento exige série contínua em intervalos de 15 minutos.")
    return clean


def load_data(source: str | Path | IO, **kwargs) -> pd.DataFrame:
    try:
        content = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content))
        header = next((row for row in reader if any(value.strip() for value in row)), [])
        # pandas silently renames duplicates and may turn excess fields into an
        # implicit index. Validate the actual CSV structure before parsing it.
        if len(set(header)) != len(header):
            raise ValueError("Há colunas duplicadas no cabeçalho do CSV.")
        for row in reader:
            if any(value.strip() for value in row) and len(row) != len(header):
                raise ValueError(f"Linha {reader.line_num}: quantidade de campos diferente do cabeçalho.")
        raw = pd.read_csv(io.StringIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"Não foi possível ler o CSV: {exc}") from exc
    return validate_data(raw, **kwargs)
