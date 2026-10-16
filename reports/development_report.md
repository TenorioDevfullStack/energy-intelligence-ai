# Relatório de desenvolvimento — 09/10/2026

MVP implementado e executado na raiz do repositório. Sem commits,
push ou PR. O notebook original, dataset e gerador foram preservados. O notebook 01
já constava como modificado no início; sua diferença no Git pertence ao estado
anterior desta rodada. SHA-256 preservado:
`EBB1C5C2CBA4A09060A0CC34D7234F9588B4F7F6068577AE2054B593B760F258`.

## 1. Arquivos criados

- `src/__init__.py`, `src/utils.py`: pacote e caminhos independentes do cwd.
- `src/data_processing.py`: leitura CSV e validação estrutural/física/temporal.
- `src/feature_engineering.py`: lags/rolling causais e perfil histórico congelado.
- `src/anomaly_detection.py`: treino, inferência, calibração e métricas.
- `src/forecasting.py`: V1, V2, baselines, seleção CV e avaliação holdout.
- `src/severity.py`: thresholds centralizados e mensagens interpretáveis.
- `src/inference.py`: carga validada, análise e previsão futura de um passo.
- `src/train.py`: CLI reproduzível de treinamento e geração de saídas.
- `app/__init__.py`, `app/api.py`, `app/charts.py`, `app/dashboard.py`.
- `tests/test_data_features.py`, `tests/test_models.py`, `tests/test_api.py`,
  `tests/test_dashboard.py`, `pytest.ini`.
- `.streamlit/config.toml`.
- `notebooks/02_mvp_model_evaluation.ipynb`: 17 células executadas.
- `docs/implementation_plan.md`.
- `docs/screenshots/dashboard.png`, `anomalies.png`, `performance.png`.
- `reports/model_metrics.json`, `reports/browser_verification.json`,
  `reports/http_verification.json`, este relatório.
- `.gitkeep` em `models/`, `data/processed/` e `data/outputs/`.

Gerados localmente e ignorados no Git: `models/anomaly_model.joblib`,
`models/forecast_model.joblib`, `models/metrics.json`,
`data/outputs/analysis_results.csv`, `anomaly_events.csv`, `forecast_results.csv`.

## 2. Arquivos alterados

`README.md`, `requirements.txt`, `.gitignore`. Nenhuma alteração desta rodada em
`01_exploratory_analysis.ipynb`, `src/generate_dataset.py` ou CSV original.

## 3. Arquitetura final

```text
CSV → validação → features → treino/TimeSeriesSplit/holdout → joblib + métricas
                                                               ↓
                                                        EnergyPipeline
                                                         ↙          ↘
                                                       API        dashboard
```

O notebook 02 documenta a mesma lógica reutilizável. A API e o dashboard são
consumidores independentes dos artefatos; nenhum deles chama fit durante inferência.

## 4. Modelos implementados

Isolation Forest V2 preserva as features contextuais originais. Contaminação
configurada em 20/2880; mediana horária e mínimos/máximos de score somente no treino.
Score = normalização de -decision_function, limitado a 0–100, sem interpretação
probabilística. Severidade considera primeiro a detecção; thresholds 40/60/80.
Mensagens mostram medições/desvios sem inventar causas.

Forecast V1: RandomForestRegressor com 200 árvores, random_state=42, features
temporais originais. V2: mesmas features mais lags 1/4/96/672 e rolling média/desvio
4/96 com shift(1), min_samples_leaf=4. Baselines: dia anterior e mediana por
hora/tipo de dia. Pipeline mantém V1/V2 e ambos os baselines; candidato principal
é **baseline_profile**, selecionado por MAE médio em CV dentro do treino.

## 5–7. Forecast V1, V2 e comparação com baselines

Treino: 2.304 medições, de 01/01 até 24/01/2026 23:45. Holdout: 576 medições, de
25/01 até 30/01/2026 23:45. Previsões de um passo (15 min), usando observações
anteriores reais; não representam previsão de seis dias sem novas medições.

| Modelo | MAE kW | RMSE kW | R² |
|---|---:|---:|---:|
| forecast_v1 | 0.5099 | 1.1495 | 0.8716 |
| forecast_v2 | 0.4806 | 1.0645 | 0.8899 |
| baseline_daily | 0.8707 | 1.7987 | 0.6855 |
| baseline_profile | 0.4126 | 0.9879 | 0.9051 |

V1 reproduz os valores históricos ≈0,51 kW / 1,15 kW / 0,872. V2 reduz o MAE em
5,7% e melhora RMSE/R², porém não supera o baseline de perfil neste demo.
A referência melhor é usada como principal; não se atribui benefício ao ML sem
superar a referência relevante.

Três janelas temporais no conjunto de treino, média ± desvio padrão:

| Modelo | MAE kW | RMSE kW | R² |
|---|---:|---:|---:|
| forecast_v1 | 0.4884 ± 0.0476 | 1.0069 ± 0.1652 | 0.8884 ± 0.0366 |
| forecast_v2 | 0.4723 ± 0.0412 | 0.9221 ± 0.0849 | 0.9084 ± 0.0204 |
| baseline_daily | 0.9153 ± 0.3167 | 1.5349 ± 0.4474 | 0.7228 ± 0.1450 |
| baseline_profile | 0.3729 ± 0.0090 | 0.8076 ± 0.1330 | 0.9279 ± 0.0258 |

Datas e métricas individuais das janelas estão em model_metrics.json. Importância
de features exibida no notebook/dashboard sem interpretação causal.

## 8. Detecção de anomalias

Histórico exploratório preservado: V1 recall 55% (11/20); V2 precision/recall/F1
80% (16/20). Esses valores são diagnóstico no dataset de ajuste.

Avaliação independente atual: precision 55,6%; recall 100%; F1 71,4%; TP 5,
FP 4, FN 0, TN 567. Matriz [[567,4],[0,5]]. Existem cinco eventos reais no holdout.
A redução da precision em relação ao número histórico reflete outro protocolo;
não é apresentada como equivalência ou garantia de generalização.
Ground truth ausente → métricas null. Rótulos parciais → avaliação apenas nas
linhas explicitamente rotuladas, indicando evaluated_rows.

## 9–10. Verificações e resultados

- `python -m pytest -q`: **53 testes aprovados**, 1 aviso de depreciação de
  Starlette sobre httpx no TestClient; nenhum teste falhou.
- `python -m src.train`: treino completo de 200 árvores executado; modelos,
  métricas e três CSVs gerados, resultados reproduzidos após revisão.
- Notebook 02: todas as 17 células executadas com a .venv; zero outputs de erro.
- HTTP real: health/docs/analyze/predict-anomaly/predict-forecast retornaram 200;
  resposta de amostra anômala com score/severidade/mensagem; forecast +15 minutos.
- Streamlit AppTest: cards, seis gráficos, filtros e tabela funcionando.
- Edge headless/Playwright: demo e três abas; erro claro para CSV com colunas
  faltando; upload compatível sem ground truth; download de 576 linhas com
  severidade; zero erros de página. Evidência em browser_verification.json.
- Revisão independente: corrigidos campos opcionais incompletos, cabeçalhos/linhas
  CSV malformados, incompatibilidade de versão mínima Streamlit e artefatos
  corrompidos/incompatíveis. Revisão final sem achados Critical/Important.
- Hash do notebook 01 inalterado; dataset/gerador sem alterações no Git;
  `git diff --check` sem erro de whitespace; `pip check` sem dependências quebradas.

Testes verificam causalidade sob mutação de valores atuais/futuros, duração de
lags com lacunas, score estável por lote, thresholds e eventos negativos, métricas,
carga sem fit, formatos, histórico anterior, validação Pydantic, API 422/503,
artefatos inválidos, gráficos com zero eventos e execução da interface.

## 11–12. Comandos para API e dashboard

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
.\.venv\Scripts\python.exe -m pytest -q
```

API e dashboard em terminais separados. API http://localhost:8000/docs;
dashboard http://localhost:8501. A aplicação pode iniciar sem modelos, mas exibe
orientação de treino; prontidão API é 503 até existirem artefatos válidos.

## 13. Limitações e condições do ambiente

Dataset sintético de 30 dias, cinco eventos de teste e somente uma instalação.
Thresholds/contaminação são políticas iniciais. Sem múltiplos passos, ingestão
contínua, autenticação ou banco de dados. Upload usa o modelo do demo; deve-se
retreinar e avaliar cada instalação. Lags V2 exigem histórico de sete dias;
histórico incompleto usa V1 explicitamente.

O sandbox negou escrita normal no disco U; gravações autorizadas foram executadas
com elevação. O disco temporário C tinha cerca de 74 MB livres e a instalação
falhou por falta de espaço; arquivos temporários foram redirecionados ao U e a
instalação foi concluída. Nenhum arquivo externo foi apagado.

Após editar módulos com Streamlit já rodando, o servidor manteve imports antigos;
reiniciar resolveu. A verificação final usa módulos e modelos atuais.

Versões do ML testadas constam em model_metrics.json. API/dashboard testados em
FastAPI 0.143.0 e Streamlit 1.65.0; ambiente Python 3.13.5. Playwright foi instalado
apenas como ferramenta local de verificação, não como dependência do produto.

## 14. Próximos passos

Dados reais/mais meses, validação por instalação, redução de falsos positivos via
calibração exclusivamente no treino, detecção de drift, armazenamento, ingestão,
controle de acesso e alertas. Forecast multi-step deve ter avaliação separada.

## 15. Sugestões de commits separados

Nenhum commit ou push realizado. Ao preparar commits, exclua as alterações
preexistentes no notebook 01, salvo decisão explícita de incluí-las.

1. `refactor: extract machine learning pipeline` — utils, validação, pacote e
   configuração pytest/requirements inicial.
2. `feat: improve forecasting with lag features` — features, forecasting.
3. `feat: add anomaly scoring and severity` — detector/severidade.
4. `feat: persist models and add inference pipeline` — inferência/CLI, .gitignore,
   diretórios e relatório de métricas.
5. `feat: add FastAPI inference API` — app/api.py e pacote app.
6. `feat: add Streamlit energy dashboard` — dashboard/charts/configuração visual.
7. `test: add pipeline and interface tests` — testes Python.
8. `docs: document MVP results and execution` — README, notebook 02, screenshots,
   plano e relatórios de desenvolvimento/verificação.

As dependências podem ser adicionadas junto ao commit da funcionalidade
correspondente. A sequência acima sugere organização para revisão; não modifica
o histórico Git automaticamente.
