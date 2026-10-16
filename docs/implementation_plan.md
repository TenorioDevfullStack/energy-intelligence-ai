# Plano do MVP — 09/10/2026

Auditoria: gerador Python, CSV com 2.880 registros/20 eventos e notebook de 59
células. Não há API, dashboard, testes ou artefatos. O notebook original tem
alterações locais; será preservado byte a byte junto com o gerador e o CSV.

Arquitetura: módulos Python compartilhados por CLI, FastAPI e Streamlit/Plotly.
Frontend separado ampliaria o escopo; manter lógica no notebook impediria reuso.
Execução no workspace atual, sem commits/push, como solicitado.

Contratos: uma instalação; timestamp local sem timezone, alinhado a 15 minutos.
Rejeitar duplicatas, NaN, infinitos, valores inválidos. Anomalias exigem potência,
tensão e fator de potência. Histórico usa lags 1/4/96/672 e rolling 4/96 shift(1).
Lacunas tornam features indisponíveis, sem alterar a duração de um lag.

Holdout final 20%; seleção por MAE médio em três janelas anteriores. Comparar V1,
V2, baseline diário e mediana por hora/tipo de dia ajustada somente no treino.
Baseline pode ser principal; V1 é fallback sem histórico. Persistir modelos dos
primeiros 80%, sem refit no holdout; identificar sobreposição de treino na análise.
IsolationForest mantém features V2 originais; medianas e normalização congeladas.
Score 0–100 é relativo, não probabilidade. Severidade considera detecção primeiro.
Dashboard: centro de operações grafite/verde, cards, filtros, seis gráficos,
eventos, upload/download, métricas e importância das features.

- [x] Testes red: validação, features sem vazamento e severidade.
- [x] Implementar processamento, features e severidade; green.
- [x] Testes red: treino, métricas, persistência e inferência.
- [x] Implementar módulos ML e CLI; executar treino e gerar artefatos/CSVs.
- [x] Testes red de API; FastAPI/Pydantic e erros 422/503.
- [x] Dashboard/Plotly; AppTest, HTTP e inspeção visual.
- [x] Notebook 02 reutilizando módulos; hash do notebook 01 intacto.
- [x] README, requirements, .gitignore e relatório final.
- [x] pytest completo, notebook, API e dashboard reais.

Verificação: python -m pytest -q; python -m src.train; uvicorn app.api:app;
streamlit run app/dashboard.py. Cobrir mutação de futuro, score independente do
lote, carregamento sem fit, persistência, dados inválidos, API e interface demo.
