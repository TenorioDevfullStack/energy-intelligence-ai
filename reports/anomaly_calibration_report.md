# Energy Intelligence AI — Etapa 2, 13/10/2026

Calibração temporal e eventos implementados sobre o MVP existente, sem commits
ou push. O forecasting e os endpoints anteriores foram preservados. O detector
persistido agora usa threshold selecionado exclusivamente na validação.

## Arquivos criados

- `src/config.py`: política de treino, recall, grade e agrupamento.
- `src/anomaly_calibration.py`: seleção reproduzível com guardas temporais.
- `src/events.py`: agregação, status, mensagens e resumo dos eventos.
- `app/event_views.py`: apresentação de eventos e comparação temporal.
- `tests/test_calibration.py`, `tests/test_events.py`,
  `tests/test_artifacts_events.py`: calibração, ausência de leakage, eventos,
  compatibilidade e consistência da política persistida.
- `notebooks/03_anomaly_calibration_and_events.ipynb`: 15 células,
  incluindo oito células de código executadas.
- `docs/anomaly_calibration_plan.md`.
- `docs/screenshots/operational_events.png`, `anomaly_calibration.png`,
  `operational_events_confirmed.png`. O último usa um CSV didático enviado,
  separado do dataset/avaliação de demonstração.
- `reports/anomaly_notebook_verification.json`,
  `reports/anomaly_runtime_verification.json` e este relatório.

## Arquivos alterados

- `src/anomaly_detection.py`: threshold persistido e métricas adicionais;
  preserva flags do corte padrão para comparação.
- `src/severity.py`: maior severidade observada no evento.
- `src/inference.py`: `analyze_with_events`, política persistida e validação
  de artefatos antigos/novos com fronteiras e hashes consistentes.
- `src/train.py`: treino/validação/holdout, comparação, persistência e novos CSVs.
- `app/api.py`: schemas e `POST /analyze/events`, prontidão com threshold/status.
- `app/dashboard.py`, `app/charts.py`: aba Eventos, timeline, controles, exportação
  e identidade do seletor de datas ao trocar a fonte/janela.
- `tests/test_api.py`, `tests/test_dashboard.py`: API e interface com/sem eventos.
- `README.md`, `reports/model_metrics.json`.

Os modelos e CSVs locais foram regenerados: `models/anomaly_model.joblib`,
`forecast_model.joblib`, `metrics.json`; `data/outputs/analysis_results.csv`,
`forecast_results.csv`, `anomaly_points.csv`, `anomaly_events.csv` e
`anomaly_thresholds.csv`. Esses artefatos continuam ignorados no Git.
`requirements.txt` e `.gitignore` já atendiam às dependências/saídas necessárias.

O checkout já tinha alterações locais e arquivos não rastreados antes desta
etapa. A lista acima se refere às mudanças após o novo pedido; diferenças
preexistentes no Git não foram apagadas nem atribuídas à Etapa 2.

## Arquitetura e períodos

```text
CSV validado → detector ajustado no treino 60%
                      ↓ perfil/escala congelados
                 Validation 20% → grade de thresholds → seleção
                      ↓ threshold persistido, sem refit
                 Holdout 20% → avaliação final
                      ↓
EnergyPipeline → análise por linha + eventos → API / Streamlit / notebook
```

| Conjunto do detector | Início | Fim | Linhas |
|---|---|---|---:|
| Treino | 01/01/2026 00:00 | 18/01/2026 23:45 | 1.728 |
| Validation | 19/01/2026 00:00 | 24/01/2026 23:45 | 576 |
| Holdout | 25/01/2026 00:00 | 30/01/2026 23:45 | 576 |

Forecasting conserva treino 80%, CV interna e holdout 20%. Sua previsão não entra
no cálculo do score nem na seleção do threshold do detector. O perfil horário,
normalização e Isolation Forest do detector usam somente os primeiros 60%.
O modelo não é refitado após calibrar, para manter a escala do corte.

## Threshold anterior e calibrado

- Modelo histórico V2, treino 80%: `decision_function < 0`, equivalente a
  **57,3000/100** na sua escala congelada.
- Detector atual, treino 60%: corte padrão equivalente a **64,6648/100**.
- Corte calibrado: **68/100**, com decisão `score >= 68`.

As duas escalas de score pertencem a modelos distintos. O efeito isolado da
calibração é comparado entre 64,6648 e 68 no mesmo detector de 60%.
Contaminação permanece fixa em 20/2880; não é extraída de labels de inferência.

A grade 0–100, passo 1, maximiza F1 com recall de validação ≥0,80. Empates
favorecem menos FP e depois maior threshold. Accuracy não decide a seleção.
Sem rótulos, sem ambas as classes ou sem corte viável, conserva-se o corte padrão
com status explícito. O score 0–100 é relativo, não uma probabilidade.

## Métricas antes/depois

| Conjunto | Versão | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Padrão, treino 60% | 71,4% | 83,3% | 76,9% | 5 | 2 | 1 | 568 |
| Validation | Calibrado, treino 60% | 100% | 83,3% | 90,9% | 5 | 0 | 1 | 570 |
| Holdout | V2 anterior, treino 80% | 55,6% | 100% | 71,4% | 5 | 4 | 0 | 567 |
| Holdout | Padrão, treino 60% | 57,1% | 80% | 66,7% | 4 | 3 | 1 | 568 |
| Holdout | Calibrado, treino 60% | 100% | 80% | 88,9% | 4 | 0 | 1 | 571 |

A evolução completa reduziu FP **4 → 0** e aumentou FN **0 → 1**. No mesmo
detector de 60%, o threshold reduziu FP **3 → 0**, mantendo FN=1/recall=80%.
A redução do recall em relação ao modelo histórico não é atribuída apenas ao
threshold: houve mudança no conjunto usado para ajustar o detector.

Holdout calibrado: matriz `[[571,0],[1,4]]`, FPR=0, specificity=1,
balanced accuracy=0,90. Validation calibrada: FPR=0, specificity=1,
balanced accuracy≈0,9167. Métricas completas e todos os thresholds estão nos
JSON/CSV. A referência histórica de 80% é reproduzida somente para comparação;
nenhuma métrica de holdout participa da seleção.

Forecast V1/V2/baselines conservaram os resultados anteriores no holdout:

| Forecast | MAE kW | RMSE kW | R² |
|---|---:|---:|---:|
| V1 | 0,5099 | 1,1495 | 0,8716 |
| V2 | 0,4806 | 1,0645 | 0,8899 |
| Dia anterior | 0,8707 | 1,7987 | 0,6855 |
| Perfil histórico, selecionado | 0,4126 | 0,9879 | 0,9051 |

## Eventos gerados, confirmados e severidade

| Escopo | Eventos | Confirmados | Suspeitas | Alto | Crítico | Duração total |
|---|---:|---:|---:|---:|---:|---:|
| Dataset completo, inclui treino | 19 | 0 | 19 | 9 | 10 | 0 min |
| Holdout | 4 | 0 | 4 | 1 | 3 | 0 min |

Não houve eventos Baixo ou Moderado nestes lotes. Todos os pontos do demo
detectados pelo corte calibrado são isolados: mantidos como suspeitas, sem criar
confirmações artificiais. Zero minutos significa duração observada entre início
e fim, que coincidem para um ponto; não é estimativa de tempo de falha física.

Gap padrão ≤15 min, persistência mínima=2. Uma medição normal interrompe o
evento mesmo com tolerância maior. Severidade é a maior dos pontos. A referência
esperada é a média das previsões; o maior desvio relativo em magnitude mantém
o sinal. Referências próximas de zero não produzem percentuais. IDs derivam do
início e mensagens descrevem dados, sem diagnosticar causas.

O exemplo didático na API/navegador tem três pontos adjacentes e um isolado:
dois eventos, um confirmado de 30 min e uma suspeita de 0 min. Ele é separado
das métricas do demo. CSV agregado contém uma linha por evento; pontos foram
preservados em `anomaly_points.csv`.

## Testes e execução

- Suíte final: **120 testes aprovados**, zero falhas. Um aviso preexistente de
  depreciação do TestClient Starlette/httpx; nenhuma dependência quebrada.
- Testes de calibração: política F1/recall/desempate, labels parciais/ausentes,
  classes insuficientes, corte inviável e rejeição de treino/holdout.
- Teste de ausência de leakage: modifica potência e labels de todo o holdout;
  threshold, grade de validação, score/perfil e previsões do detector não mudam.
- Eventos: consecutivos/isolados, grupos separados, normalidade entre pontos,
  gaps/tolerância, duração, severidade, referências zero, desvios mistos e IDs.
- Artefatos/API: legado, metadados incompletos, hashes inconsistentes, política
  persistida, overrides, prontidão 503 e input inválido 422.
- Dashboard: eventos/estado vazio, políticas persistidas 1500 pontos/0,5 min/2880
  min, filtros e preservação das datas em ações não relacionadas à janela.
- Treino de 200 árvores executado, modelos/grade/CSVs gerados.
- Notebook 03: 15 células, oito células de código executadas, zero outputs de
  erro; curva PR de Validation com threshold selecionado.
- Verificação real da API e Edge/Playwright: evidências e screenshots nos arquivos
  `anomaly_runtime_verification.json` e `docs/screenshots/`.
- Revisão independente: corrigidos metadados incompletos, política divergente
  entre consumidores e limites da interface; revisão final sem achados relevantes.
- `pip check` sem dependências quebradas; `git diff --check` sem erros.

```powershell
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

API/documentação: http://localhost:8000/docs. Dashboard: http://localhost:8501.
`POST /analyze/events` recebe `records`, histórico opcional e overrides de
persistência/gap. Retorna `summary`, `events` e métricas por ponto quando há labels.
`/analyze` mantém análise por linha. A política omitida vem do modelo persistido,
coerente com treino/dashboard.

## Preservação, limitações e próximos passos

Hashes de notebook 01, notebook 02, gerador e CSV permaneceram iguais ao início
da Etapa 2. Notebook original SHA-256:
`ebb1c5c2cba4a09060a0cc34d7234f9588b4f7f6068577ae2054b593b760f258`.
Arquivos temporários de verificação ficam em `.runtime-tmp/`, ignorada no Git.

Limitações: uma instalação sintética de 30 dias, seis positivos na Validation
e cinco no Holdout. O aumento de FN é explícito. Recall mínimo é uma restrição
da Validation, não uma garantia para dados futuros. Confirmado significa
persistência, não diagnóstico. Eventos são locais ao lote, sem banco ou
continuidade entre requisições; filtros preservam eventos completos dentro da
fonte/janela elegível. Neste Windows, o sandbox bloqueia pipes usados pelos
modelos; serviços/testes foram executados fora dele, com autorização da tarefa.

Próximos passos: validar a política em mais períodos/instalações, medir custos
de FP/FN, avaliar persistência com eventos reais, monitorar drift e implementar
armazenamento/continuidade de eventos se necessário. Não ajustar retrospectivamente
o threshold com base neste holdout; usar uma nova validação para novas políticas.

## Sugestões de commits separados

1. `feat: calibrate anomaly detection threshold`
2. `feat: add operational anomaly events`
3. `feat: add event analytics to API and dashboard`
4. `test: add anomaly calibration and event tests`
5. `docs: document anomaly calibration results`

Nenhum commit ou push foi realizado.
