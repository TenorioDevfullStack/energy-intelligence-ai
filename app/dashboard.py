"""Energy operations dashboard. ML remains in src, never in this interface."""
import io
import hashlib
import sys
from pathlib import Path

# Streamlit executes scripts by path, including when launched outside the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import charts
from app.event_views import render_events, render_calibration_comparison
from app.resources import load_dashboard_resources, resource_signature
from src.anomaly_detection import evaluate_anomaly_model
from src.data_processing import load_data
from src.inference import ModelArtifactError
from src.severity import SEVERITY_ORDER
from src.utils import DATASET_PATH, MODEL_DIR
from src.events import build_anomaly_events
from src.config import MAX_EVENT_GAP_MINUTES, MIN_EVENT_POINTS

st.set_page_config(page_title="Energy Intelligence AI", page_icon="⚡", layout="wide")

CSS = """
<style>
.stApp {background:#101917;color:#edf3ed;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stHeader"] {background:#101917;}
.block-container {padding-top:4.2rem;padding-bottom:3rem;max-width:1600px;}
h1,h2,h3 {font-family:Bahnschrift,'Trebuchet MS',sans-serif !important;letter-spacing:-.035em;}
h1 {font-size:2.4rem !important;font-weight:600 !important;}
h3 {font-size:1.05rem !important;letter-spacing:0 !important;}
[data-testid="stSidebar"] {background:#14201b;border-right:1px solid #2a3c31;}
[data-testid="stMetric"] {background:#1a2721;border:1px solid #304335;border-top:2px solid #b5ed6b;
  border-radius:10px;padding:18px 16px;min-height:115px;}
[data-testid="stMetricLabel"] {color:#afc4b7;}
[data-testid="stMetricValue"] {font-family:Bahnschrift,monospace;font-size:1.65rem;}
[data-testid="stTabs"] button {font-family:Bahnschrift,sans-serif;font-size:1rem;}
.eyebrow {color:#b5ed6b;font-family:Bahnschrift,monospace;letter-spacing:.16em;font-size:.72rem;}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:26px;}
.mark {background:#b5ed6b;color:#14201b;padding:8px 12px;font-family:Bahnschrift,monospace;
 font-size:1.2rem;font-weight:700;border-radius:6px;}
.brand-text {font-family:Bahnschrift,sans-serif;line-height:1.25;font-weight:600;font-size:.95rem;}
.footer {border-top:1px solid #2c3f33;padding-top:18px;margin-top:24px;color:#8ca594;font-size:.8rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
MODEL_LABELS = {"forecast_v1": "Forecast V1", "forecast_v2": "Forecast V2",
                "baseline_daily": "Dia anterior", "baseline_profile": "Perfil histórico"}


@st.cache_data(show_spinner=False, max_entries=4, ttl=900)
def analyze_csv(content: bytes, model_dir: str, revision: tuple, demo_content: bytes) -> pd.DataFrame:
    data = load_data(io.BytesIO(content), require_electrical=True)
    return load_dashboard_resources(model_dir, revision, demo_content).pipeline.analyze(data)


def render() -> None:
    try:
        demo_content = DATASET_PATH.read_bytes()
    except OSError:
        st.error("Dataset de demonstração indisponível. Restaure data/raw/energy_consumption_30days.csv.")
        return
    with st.sidebar:
        st.markdown('<div class="brand"><span class="mark">E/</span><span class="brand-text">ENERGY<br>INTELLIGENCE AI</span></div>', unsafe_allow_html=True)
        st.caption("CENTRO DE ANÁLISE ENERGÉTICA")
        st.divider()
        st.subheader("Fonte de dados")
        upload = st.file_uploader("Upload CSV", type=["csv"], help="Uma instalação; medições de 15 minutos.", key="csv_upload")
        st.caption("Colunas: timestamp, active_power_kw, voltage_v, power_factor. is_anomaly é opcional.")
        st.download_button("Baixar CSV de demonstração", demo_content, "energy_demo.csv", "text/csv", width="stretch")
    st.markdown('<div class="eyebrow">ENERGIA / DADOS / INTELIGÊNCIA</div>', unsafe_allow_html=True)
    st.title("Energy Intelligence AI")
    st.caption("Inteligência aplicada ao consumo e comportamento energético")
    try:
        revision = resource_signature(MODEL_DIR)
        content = upload.getvalue() if upload is not None else demo_content
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("O CSV deve ter até 20 MB.")
        # Validate before any startup work; uploads are never used for training.
        load_data(io.BytesIO(content), require_electrical=True)
        resources = load_dashboard_resources(str(MODEL_DIR), revision, demo_content)
        pipeline, metrics = resources.pipeline, resources.metrics
        results = analyze_csv(content, str(MODEL_DIR), revision, demo_content)
    except (FileNotFoundError, OSError, ModelArtifactError) as exc:
        st.error(f"Não foi possível carregar os artefatos: {exc}")
        st.code("python -m src.train")
        return
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"CSV incompatível: {exc}")
        return
    demo = upload is None
    if resources.prepared_demo:
        st.info("Modo demo: modelos da demonstração preparados automaticamente e mantidos em cache. "
                "Uploads usam esses modelos, sem novo treinamento.")
    st.caption("DEMONSTRAÇÃO · dados sintéticos de janeiro/2026" if demo else "CSV ENVIADO · análise com os modelos persistidos da demonstração")
    with st.sidebar:
        st.divider()
        st.subheader("Janela de análise")
        only_holdout = st.checkbox("Somente após o treino", value=demo,
            help="Exclui observações que se sobrepõem ao período usado para ajustar os modelos.")
        eligible = results.loc[~results.model_training_overlap] if only_holdout else results
        if eligible.empty:
            st.info("Nenhuma medição após o período de treino. Desative o filtro para analisar o CSV.")
            return
        start, end = eligible.timestamp.min().date(), eligible.timestamp.max().date()
        period_context = (hashlib.sha256(content).hexdigest(), only_holdout, str(MODEL_DIR), revision)
        # A keyed widget preserves old dates even when its default changes.
        # Reset only when the dataset or eligible window changes; keep manual
        # date choices on other interactions such as event severity filters.
        period_key = "period_" + hashlib.sha256(repr(period_context).encode()).hexdigest()
        period = st.date_input("Período", value=(start, end), min_value=start, max_value=end,
            format="DD/MM/YYYY", key=period_key)
        selected_severities = st.multiselect("Severidades na tabela de eventos", SEVERITY_ORDER, default=SEVERITY_ORDER)
        st.subheader("Agrupamento de eventos")
        event_policy = pipeline.anomaly.get("event_policy", {})
        saved_points = int(event_policy.get("min_event_points", MIN_EVENT_POINTS))
        saved_gap = float(event_policy.get("max_gap_minutes", MAX_EVENT_GAP_MINUTES))
        min_event_points = st.number_input("Pontos para confirmar", min_value=1, max_value=max(1000, saved_points),
            value=saved_points, step=1)
        max_event_gap = st.number_input("Intervalo máximo entre pontos (min)", min_value=min(1., saved_gap), max_value=max(1440., saved_gap),
            value=saved_gap, step=min(1., saved_gap))
        st.divider()
        st.caption("RESOLUÇÃO: 15 MINUTOS")
        st.caption(f"Modelo: {MODEL_LABELS[pipeline.forecast['selected']]}")
        st.caption(f"Treinado até {pd.Timestamp(metrics['train_end']):%d/%m/%Y %H:%M}")
    if len(period) != 2:
        st.info("Selecione a data inicial e a data final.")
        return
    view = eligible.loc[eligible.timestamp.dt.date.between(period[0], period[1])].copy()
    if view.empty:
        st.info("Não há medições no período selecionado.")
        return
    if view.model_training_overlap.any():
        st.info("Este período inclui dados usados no treino. Seus indicadores são descritivos; a avaliação independente está na aba Performance.")
    if not demo:
        st.info("Modelos ajustados ao dataset sintético de demonstração. Valide e retreine para o padrão de uma instalação real antes do uso operacional.")
    fallbacks = int(view.forecast_source.eq("forecast_v1_fallback").sum())
    if fallbacks:
        history_requirement = "sete dias" if pipeline.forecast["selected"] == "forecast_v2" else "24 horas"
        st.caption(f"{fallbacks} previsões usam V1 por histórico incompleto para o candidato selecionado ({history_requirement}).")
    last = view.iloc[-1]
    events = view.loc[view.is_anomaly_prediction.eq(1)]
    critical = int(events.severity.eq("Crítico").sum())
    mae = float(view.forecast_error_kw.abs().mean())
    st.caption(f"{view.timestamp.min():%d/%m/%Y %H:%M} — {view.timestamp.max():%d/%m/%Y %H:%M} · {len(view):,} medições · horários locais da instalação")
    cards = st.columns(6)
    values = [("Última potência", f"{last.active_power_kw:.2f} kW"),
        ("Previsão no instante", f"{last.expected_power_kw:.2f} kW"),
        ("Pontos anômalos", str(len(events))), ("Pontos críticos", str(critical)),
        ("MAE do período", f"{mae:.2f} kW"),
        ("Status analítico", "Atenção" if bool(last.is_anomaly_prediction) else "Normal")]
    for column, (label, value) in zip(cards, values):
        column.metric(label, value)
    st.caption(f"Última medição: {last.timestamp:%d/%m/%Y %H:%M}. O status descreve o detector na última amostra.")
    overview, anomalies, operational, performance = st.tabs(["Visão geral", "Anomalias e eventos", "Eventos", "Performance"])
    config = {"displaylogo": False, "scrollZoom": False}
    with overview:
        st.subheader("Curva de carga")
        st.caption("Potência real × previsão de um passo; observações anteriores alimentam o histórico.")
        st.plotly_chart(charts.load_curve(view), width="stretch", config=config)
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("Forecast · erro de previsão")
            st.plotly_chart(charts.forecast_error(view), width="stretch", config=config)
        with right:
            st.subheader("Perfil médio de carga")
            st.plotly_chart(charts.hourly_profile(view), width="stretch", config=config)
        st.download_button("Exportar análise do período", view.to_csv(index=False).encode("utf-8-sig"),
            "analysis_results.csv", "text/csv", key="download_analysis")
    with anomalies:
        st.subheader("Eventos ao longo do tempo")
        st.plotly_chart(charts.anomaly_timeline(view), width="stretch", config=config)
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("Anomaly Score")
            st.plotly_chart(charts.score_timeline(view), width="stretch", config=config)
        with right:
            st.subheader("Distribuição de severidade")
            st.plotly_chart(charts.severity_distribution(view), width="stretch", config=config)
        st.caption("Anomaly score 0–100 não representa probabilidade. Escala relativa congelada no treino; severidade é atribuída somente a pontos detectados.")
        st.subheader("Eventos recentes")
        listed = events.loc[events.severity.isin(selected_severities)].sort_values("timestamp", ascending=False)
        display = listed[["timestamp", "active_power_kw", "potencia_mediana_hora", "anomaly_score", "severity", "event_message"]].rename(columns={
            "timestamp": "Horário", "active_power_kw": "Potência (kW)",
            "potencia_mediana_hora": "Típica (kW)", "anomaly_score": "Score",
            "severity": "Severidade", "event_message": "Mensagem"})
        st.dataframe(display, hide_index=True, width="stretch")
        if listed.empty:
            st.info("Nenhum evento corresponde aos filtros selecionados.")
        st.download_button("Exportar pontos anômalos", listed.to_csv(index=False).encode("utf-8-sig"), "anomaly_points.csv", "text/csv")
    with operational:
        aggregated = build_anomaly_events(eligible, max_gap_minutes=max_event_gap, min_event_points=min_event_points)
        if not aggregated.empty:
            window_start = pd.Timestamp(period[0])
            window_end = pd.Timestamp(period[1]) + pd.Timedelta(days=1)
            aggregated = aggregated.loc[aggregated.end_timestamp.ge(window_start) & aggregated.start_timestamp.lt(window_end)]
        render_events(view, aggregated, selected_severities)
    with performance:
        render_calibration_comparison(metrics)
        st.subheader("Avaliação independente dos modelos")
        st.caption(f"Holdout do treinamento: {pd.Timestamp(metrics['holdout_start']):%d/%m/%Y} a {pd.Timestamp(metrics['holdout_end']):%d/%m/%Y} · {metrics['holdout_rows']} amostras. Estes resultados não mudam com os filtros ou upload.")
        table = pd.DataFrame(metrics["forecast_holdout"]).T.rename(columns={"mae_kw":"MAE (kW)","rmse_kw":"RMSE (kW)","r2":"R²"})
        st.dataframe(table.rename(index=MODEL_LABELS).round(4), width="stretch")
        st.caption(f"Candidato principal: {MODEL_LABELS[metrics['forecast_selection']]}. Seleção por MAE médio em três janelas anteriores ao holdout.")
        cv = pd.DataFrame({name:{f"{metric} · {stat}":value for metric,stats in scores.items() for stat,value in stats.items()}
            for name,scores in metrics["forecast_cv"].items()}).T
        with st.expander("Validação temporal: médias e desvios"):
            st.dataframe(cv.rename(index=MODEL_LABELS).round(4), width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Isolation Forest V2 · holdout")
            scores = metrics["anomaly_holdout"]
            if scores is not None:
                a,b,c = st.columns(3)
                a.metric("Precision", f"{scores['precision']:.0%}")
                b.metric("Recall", f"{scores['recall']:.0%}")
                c.metric("F1", f"{scores['f1']:.0%}")
                st.caption(f"TP {scores['true_positives']} · FP {scores['false_positives']} · FN {scores['false_negatives']} · TN {scores['true_negatives']}")
            else:
                st.info("Sem ground truth no treinamento para calcular métricas supervisionadas.")
            period_scores = evaluate_anomaly_model(view, view.is_anomaly_prediction)
            with st.expander("Métricas supervisionadas do período selecionado"):
                if period_scores is None:
                    st.info("CSV sem is_anomaly: precision, recall e F1 indisponíveis.")
                else:
                    st.json(period_scores)
        with right:
            st.subheader("Importância das features · Forecast V2")
            importance = pd.Series(metrics["feature_importance_v2"]).sort_values(ascending=False).head(10)
            st.bar_chart(importance, color=charts.GREEN)
            st.caption("Importância interna do Random Forest; não demonstra causalidade.")
    st.markdown('<div class="footer">ENERGY INTELLIGENCE AI · MVP 0.3 &nbsp; / &nbsp; Calibração temporal · eventos operacionais</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    render()
