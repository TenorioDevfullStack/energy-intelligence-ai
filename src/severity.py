"""Operational severity rules; these are configurable policy, not diagnosis."""
import math

SEVERITY_THRESHOLDS = ((40., "Baixo"), (60., "Moderado"), (80., "Alto"))
SEVERITY_ORDER = ["Baixo", "Moderado", "Alto", "Crítico"]


def aggregate_event_severity(severities) -> str:
    """An event takes the highest observed point severity, never a new diagnosis."""
    values = list(severities)
    if not values or any(value not in SEVERITY_ORDER for value in values):
        raise ValueError("Evento exige severidades de pontos detectados válidas.")
    return max(values, key=SEVERITY_ORDER.index)


def classify_severity(score: float, detected: bool) -> str:
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Score deve estar entre 0 e 100.")
    if not detected:
        return "Normal"
    return next((label for limit, label in SEVERITY_THRESHOLDS if score < limit), "Crítico")


def event_message(power: float, typical: float, score: float, severity: str) -> str:
    if severity == "Normal":
        return "Operação dentro do padrão histórico do detector."
    deviation = f"{(power / typical - 1) * 100:+.1f}%" if typical > 1e-6 else "indisponível (referência próxima de zero)"
    return (f"Comportamento anômalo — severidade {severity}. Potência observada: {power:.2f} kW; "
            f"potência típica do horário: {typical:.2f} kW; desvio relativo: {deviation}; "
            f"Anomaly Score: {score:.1f}/100.")
