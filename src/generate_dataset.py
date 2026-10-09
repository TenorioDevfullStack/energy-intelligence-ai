import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# --------------------------------------------------
# Configurações
# --------------------------------------------------

DAYS = 30
FREQUENCY = "15min"

start_date = "2026-01-01 00:00:00"

timestamps = pd.date_range(
    start=start_date,
    periods=DAYS * 24 * 4,
    freq=FREQUENCY
)

# --------------------------------------------------
# Gerando perfil de consumo
# --------------------------------------------------

data = []

for timestamp in timestamps:

    hour = timestamp.hour
    weekday = timestamp.weekday()

    # Perfil básico de carga
    if 0 <= hour < 6:
        base_power = 1.5

    elif 6 <= hour < 8:
        base_power = 3.5

    elif 8 <= hour < 12:
        base_power = 7.0

    elif 12 <= hour < 14:
        base_power = 5.0

    elif 14 <= hour < 18:
        base_power = 8.0

    elif 18 <= hour < 22:
        base_power = 10.0

    else:
        base_power = 4.0

    # Redução aos finais de semana
    if weekday >= 5:
        base_power *= 0.65

    # Pequena variação natural
    active_power_kw = np.random.normal(
        loc=base_power,
        scale=base_power * 0.08
    )

    active_power_kw = max(active_power_kw, 0.3)

    # Tensão simulada
    voltage = np.random.normal(
        loc=127,
        scale=1.5
    )

    # Fator de potência
    power_factor = np.random.normal(
        loc=0.94,
        scale=0.02
    )

    power_factor = np.clip(
        power_factor,
        0.80,
        1.0
    )

    # Corrente aproximada
    current = (
        active_power_kw * 1000
    ) / (
        voltage * power_factor
    )

    # Energia consumida em 15 minutos
    energy_kwh = active_power_kw * 0.25

    data.append(
        [
            timestamp,
            voltage,
            current,
            active_power_kw,
            power_factor,
            energy_kwh
        ]
    )

# --------------------------------------------------
# Criando DataFrame
# --------------------------------------------------

df = pd.DataFrame(
    data,
    columns=[
        "timestamp",
        "voltage_v",
        "current_a",
        "active_power_kw",
        "power_factor",
        "energy_kwh"
    ]
)

# --------------------------------------------------
# Inserindo anomalias propositalmente
# --------------------------------------------------

num_anomalies = 20

anomaly_indices = np.random.choice(
    df.index,
    size=num_anomalies,
    replace=False
)

df.loc[
    anomaly_indices,
    "active_power_kw"
] *= np.random.uniform(
    1.8,
    2.8,
    size=num_anomalies
)

# Recalcular corrente após anomalia
df["current_a"] = (
    df["active_power_kw"] * 1000
) / (
    df["voltage_v"] * df["power_factor"]
)

# Recalcular energia
df["energy_kwh"] = (
    df["active_power_kw"] * 0.25
)

# Marcação para sabermos a resposta verdadeira
df["is_anomaly"] = 0

df.loc[
    anomaly_indices,
    "is_anomaly"
] = 1

# --------------------------------------------------
# Arredondamento
# --------------------------------------------------

columns_to_round = [
    "voltage_v",
    "current_a",
    "active_power_kw",
    "power_factor",
    "energy_kwh"
]

df[columns_to_round] = df[
    columns_to_round
].round(3)

# --------------------------------------------------
# Salvando arquivo
# --------------------------------------------------

output_dir = Path("data/raw")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir /
    "energy_consumption_30days.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("Dataset criado com sucesso!")
print()
print(f"Registros: {len(df)}")
print(f"Anomalias inseridas: {df['is_anomaly'].sum()}")
print(f"Arquivo: {output_file}")

print()
print(df.head())