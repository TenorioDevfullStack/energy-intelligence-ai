"""Plotly views without ML logic; shared by dashboard and notebook."""
import pandas as pd
import plotly.graph_objects as go

from src.severity import SEVERITY_ORDER

GREEN = "#b5ed6b"
BLUE = "#78b9ed"
COLORS = {"Baixo": "#82b7af", "Moderado": "#e4c464", "Alto": "#e99762", "Crítico": "#ef7787"}


def _style(fig: go.Figure, y_title: str = "", height: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Bahnschrift, sans-serif", color="#c8d3d0", size=12),
        margin=dict(l=12, r=16, t=35, b=20), hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0), yaxis_title=y_title)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#263531", zeroline=False)
    return fig


def load_curve(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.timestamp, y=rows.active_power_kw, name="Potência real",
        mode="lines", line=dict(color=GREEN, width=1.8)))
    fig.add_trace(go.Scatter(x=rows.timestamp, y=rows.expected_power_kw, name="Potência prevista",
        mode="lines", line=dict(color=BLUE, width=1.8, dash="dot")))
    return _style(fig, "Potência · kW", 380)


def anomaly_timeline(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rows.timestamp, y=rows.active_power_kw, name="Carga observada",
        mode="lines", line=dict(color="#607a70", width=1)))
    for severity in SEVERITY_ORDER:
        events = rows.loc[rows.is_anomaly_prediction.eq(1) & rows.severity.eq(severity)]
        fig.add_trace(go.Scatter(x=events.timestamp, y=events.active_power_kw, name=severity,
            mode="markers", marker=dict(color=COLORS[severity], size=10, symbol="diamond"),
            customdata=events[["anomaly_score", "potencia_mediana_hora"]].to_numpy(),
            hovertemplate="%{x}<br>%{y:.2f} kW<br>Score: %{customdata[0]:.1f}<br>Típica: %{customdata[1]:.2f} kW<extra>%{fullData.name}</extra>"))
    return _style(fig, "Potência · kW")


def score_timeline(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rows.timestamp, y=rows.anomaly_score, name="Anomaly Score",
        mode="lines", line=dict(color=BLUE, width=1.5)))
    detected = rows.loc[rows.is_anomaly_prediction.eq(1)]
    fig.add_trace(go.Scatter(x=detected.timestamp, y=detected.anomaly_score, name="Evento detectado",
        mode="markers", marker=dict(color="#ef7787", size=7)))
    _style(fig, "Índice · 0–100")
    fig.update_yaxes(range=[0, 105])
    return fig


def severity_distribution(rows: pd.DataFrame) -> go.Figure:
    counts = rows.loc[rows.is_anomaly_prediction.eq(1), "severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[COLORS[x] for x in counts.index],
        text=counts.values, textposition="outside", name="Eventos"))
    _style(fig, "Eventos", 300)
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(dtick=1, rangemode="tozero")
    return fig


def forecast_error(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=rows.timestamp, y=rows.forecast_error_kw,
        name="Real − prevista", mode="lines", line=dict(color="#e4c464", width=1.4)))
    fig.add_hline(y=0, line_color="#53645d", line_dash="dot")
    return _style(fig, "Erro · kW")


def hourly_profile(rows: pd.DataFrame) -> go.Figure:
    profile = rows.groupby(rows.timestamp.dt.hour).active_power_kw.mean()
    fig = go.Figure(go.Bar(x=profile.index, y=profile.values,
        marker_color=GREEN, name="Potência média"))
    _style(fig, "Potência média · kW", 300)
    fig.update_xaxes(title="Hora local", dtick=3)
    return fig


def operational_events_timeline(rows: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """Show event spans over the observed load; fixed trace count for large batches."""
    fig = go.Figure(go.Scatter(x=rows.timestamp, y=rows.active_power_kw, name="Carga observada",
        mode="lines", line=dict(color="#607a70", width=1)))
    for (severity, status), selected in events.groupby(["severity", "status"], sort=False):
        x, y, details = [], [], []
        for event in selected.itertuples():
            info = [event.event_id, event.duration_minutes, event.number_of_points,
                    event.max_anomaly_score, "Confirmado" if status == "confirmed" else "Suspeita"]
            x.extend([event.start_timestamp, event.end_timestamp, None])
            y.extend([event.max_active_power_kw, event.max_active_power_kw, None])
            details.extend([info, info, [None] * len(info)])
        fig.add_trace(go.Scatter(x=x, y=y, name=f"{severity} · {'confirmado' if status == 'confirmed' else 'suspeita'}",
            mode="lines+markers", line=dict(color=COLORS[severity], width=5),
            marker=dict(color=COLORS[severity], size=9, symbol="diamond"), customdata=details,
            hovertemplate="%{customdata[0]}<br>%{x}<br>Máxima: %{y:.2f} kW<br>Duração: %{customdata[1]} min"
                          "<br>Pontos: %{customdata[2]}<br>Score máximo: %{customdata[3]:.1f}<br>%{customdata[4]}<extra>%{fullData.name}</extra>"))
    return _style(fig, "Potência · kW", 380)
