from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from app.charts import load_curve, anomaly_timeline, score_timeline, severity_distribution, forecast_error, hourly_profile
from src.inference import EnergyPipeline
from src.data_processing import load_data
from src.utils import DATASET_PATH, MODEL_DIR, ROOT


def test_all_charts_support_empty_anomaly_sets():
    rows = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="15min"),
        "active_power_kw": [1, 2, 1, 2], "expected_power_kw": [1, 1, 1, 1],
        "is_anomaly_prediction": [0]*4, "anomaly_score": [5.]*4,
        "severity": ["Normal"]*4, "forecast_error_kw": [0., 1., 0., 1.],
        "potencia_mediana_hora": [1.]*4})
    for builder in (load_curve, anomaly_timeline, score_timeline, severity_distribution, forecast_error, hourly_profile):
        figure = builder(rows)
        assert len(figure.data) >= 1
        assert figure.layout.height >= 250


def test_dashboard_demo_and_filters():
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Energy Intelligence AI"
    assert len(app.metric) >= 6
    assert len(app.get("plotly_chart")) >= 6
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception
    assert len(app.dataframe) >= 1
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception


def test_period_resets_when_training_filter_changes():
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    data = load_data(DATASET_PATH, require_electrical=True)
    period_key = app.sidebar.date_input[0].key
    app.sidebar.date_input[0].set_value((data.timestamp.iloc[-1].date(),) * 2).run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert app.sidebar.date_input[0].value == (data.timestamp.iloc[-1].date(),) * 2
    assert app.sidebar.date_input[0].key == period_key
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception
    assert app.sidebar.date_input[0].key != period_key
    assert app.sidebar.date_input[0].value == (
        data.timestamp.min().date(), data.timestamp.max().date())


def test_period_resets_when_dataset_changes(tmp_path, monkeypatch):
    import src.utils
    demo_file = tmp_path / "demo.csv"
    demo_file.write_bytes(DATASET_PATH.read_bytes())
    monkeypatch.setattr(src.utils, "DATASET_PATH", demo_file)
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    data = load_data(demo_file, require_electrical=True)
    app.sidebar.date_input[0].set_value((data.timestamp.iloc[-1].date(),) * 2).run()
    future = data.tail(8).copy()
    future.timestamp += pd.Timedelta(days=40)
    future.to_csv(demo_file, index=False)
    app.run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) >= 6
    assert app.sidebar.date_input[0].value == (
        future.timestamp.min().date(), future.timestamp.max().date())


def test_dashboard_operational_events_tab_has_cards_table_and_timeline():
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception
    assert "Eventos" in [tab.label for tab in app.tabs]
    assert {"Eventos confirmados", "Duração total anômala"}.issubset({metric.label for metric in app.metric})
    assert len(app.get("plotly_chart")) >= 7
    assert any({"ID", "Início", "Fim", "Status"}.issubset(table.value.columns) for table in app.dataframe)


def test_dashboard_empty_operational_events(tmp_path, monkeypatch):
    import src.utils
    dataset = tmp_path / "healthy.csv"
    pd.DataFrame({"timestamp": ["2026-02-01T12:00:00"], "active_power_kw": [5.],
                  "voltage_v": [127.], "power_factor": [.94]}).to_csv(dataset, index=False)
    monkeypatch.setattr(src.utils, "DATASET_PATH", dataset)
    app = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=60).run()
    assert not app.exception and not app.error
    assert "Eventos" in [tab.label for tab in app.tabs]
    assert any("Nenhum evento operacional" in item.value for item in app.info)
    assert next(metric for metric in app.metric if metric.label == "Eventos confirmados").value == "0"


def test_operational_event_chart_supports_no_events_and_intervals():
    from src.events import build_anomaly_events
    from app.charts import operational_events_timeline
    rows = pd.DataFrame({"timestamp": pd.date_range("2026-02-01", periods=3, freq="15min"),
        "active_power_kw": 18., "expected_power_kw": 9., "is_anomaly_prediction": 0,
        "anomaly_score": 90., "severity": "Normal"})
    assert len(operational_events_timeline(rows, build_anomaly_events(rows)).data) >= 1
    rows["is_anomaly_prediction"] = 1
    rows["severity"] = "Crítico"
    figure = operational_events_timeline(rows, build_anomaly_events(rows))
    assert len(figure.data) >= 2
