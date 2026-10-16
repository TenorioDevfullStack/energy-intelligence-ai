"""Rendering of operational events and calibration evidence; no fitting logic."""
import pandas as pd
import streamlit as st

from app import charts
from src.events import summarize_anomaly_events


def render_events(rows: pd.DataFrame, events: pd.DataFrame, selected_severities: list[str]) -> None:
    st.subheader("Eventos operacionais")
    summary = summarize_anomaly_events(events)
    cards = st.columns(4)
    for column, (label, value) in zip(cards, [
            ("Eventos detectados", str(summary["total_events"])),
            ("Eventos confirmados", str(summary["confirmed_events"])),
            ("Eventos críticos", str(summary["critical_events"])),
            ("Duração total anômala", f"{summary['total_duration_minutes']:g} min")]):
        column.metric(label, value)
    st.caption("Confirmado indica persistência de pontos consecutivos; não confirma causa física. "
               "Pontos isolados permanecem como suspeitas. Duração = fim − início; um ponto tem 0 min.")
    st.plotly_chart(charts.operational_events_timeline(rows, events), width="stretch", config={"displaylogo": False})
    if events.empty:
        st.info("Nenhum evento operacional no período selecionado.")
    st.caption("Eventos que intersectam a janela selecionada; a tabela conserva início, fim e duração completos de cada evento.")
    listed = events.loc[events.severity.isin(selected_severities)].sort_values("start_timestamp", ascending=False)
    display = listed.rename(columns={"event_id": "ID", "start_timestamp": "Início", "end_timestamp": "Fim",
        "duration_minutes": "Duração (min)", "number_of_points": "Pontos", "max_active_power_kw": "Potência máxima (kW)",
        "max_anomaly_score": "Score máximo", "severity": "Severidade", "status": "Status", "message": "Mensagem"})
    display["Status"] = display["Status"].map({"suspected": "Suspeita", "confirmed": "Confirmado"})
    st.dataframe(display[["ID", "Início", "Fim", "Duração (min)", "Pontos", "Potência máxima (kW)",
                          "Score máximo", "Severidade", "Status", "Mensagem"]], hide_index=True, width="stretch")
    if not events.empty and listed.empty:
        st.info("Nenhum evento corresponde às severidades selecionadas.")
    st.download_button("Exportar eventos operacionais", listed.to_csv(index=False).encode("utf-8-sig"),
                       "anomaly_events.csv", "text/csv", key="download_operational_events")


def render_calibration_comparison(metrics: dict) -> None:
    comparison = metrics.get("anomaly_comparison")
    if comparison is None:
        st.info("Artefatos anteriores à calibração. Execute python -m src.train para gerar a comparação temporal.")
        return
    st.subheader("Detecção V2 original × calibrada")
    periods = metrics["anomaly_split"]
    st.caption(f"Detector: treino {pd.Timestamp(periods['train_start']):%d/%m/%Y} a "
               f"{pd.Timestamp(periods['train_end']):%d/%m/%Y}; Validation "
               f"{pd.Timestamp(periods['validation_start']):%d/%m/%Y} a {pd.Timestamp(periods['validation_end']):%d/%m/%Y}; "
               f"Holdout {pd.Timestamp(periods['holdout_start']):%d/%m/%Y} a {pd.Timestamp(periods['holdout_end']):%d/%m/%Y}.")
    calibration = metrics["anomaly_threshold_calibration"]
    threshold = calibration["selected_threshold"]
    st.caption(f"Threshold original (mesmo detector): {calibration['default_threshold']:.4f}/100. "
               "Anomaly score 0–100 não representa probabilidade.")
    if threshold is None:
        st.info(f"Calibração indisponível ({calibration['status']}); mantido o corte padrão do detector.")
    else:
        st.caption(f"Threshold selecionado: {threshold:.1f}/100 · recall mínimo na Validation: "
                   f"{calibration['min_recall']:.0%}. Holdout não participa da seleção. Score não é probabilidade.")
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1", "false_positives": "FP",
              "false_negatives": "FN", "false_positive_rate": "FPR", "specificity": "Specificity",
              "balanced_accuracy": "Balanced accuracy"}
    columns = st.columns(2)
    for column, (period, title) in zip(columns, [("validation", "Validation · seleção"), ("holdout", "Holdout · avaliação final")]):
        with column:
            st.markdown(f"**{title}**")
            values = comparison[period]
            if values["default"] is None:
                st.info("Sem ground truth para comparação supervisionada.")
            else:
                table = pd.DataFrame({"Corte padrão · treino 60%": values["default"],
                                      "Corte calibrado · treino 60%": values["calibrated"]}).T
                st.dataframe(table[list(labels)].rename(columns=labels).round(4), width="stretch")
    legacy = (metrics.get("anomaly_legacy") or {}).get("holdout")
    if legacy is not None:
        st.caption(f"Referência V2 anterior, treinada em 80%: Holdout Precision {legacy['precision']:.1%}, "
                   f"Recall {legacy['recall']:.1%}, F1 {legacy['f1']:.1%}, FP {legacy['false_positives']}, "
                   f"FN {legacy['false_negatives']}. O treino é diferente; a comparação acima isola apenas o threshold.")
