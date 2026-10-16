# Energy Intelligence AI — relatório final da Etapa 2

Data: 16/10/2026. Preparação para publicação concluída e verificada localmente.
Nenhum commit, push ou deploy público foi executado. A publicação no Streamlit
Community Cloud continua sendo uma ação do proprietário do repositório.

## 1. Implementação e preservação

A Etapa 2 existente foi auditada e validada, preservando módulos, arquitetura,
API, dashboard e forecasting. A lacuna de inicialização de um clone sem modelos
foi corrigida. O dashboard agora prepara o pequeno demo automaticamente, mantém
os modelos em cache e aceita uploads sem executar treinamento com esses arquivos.

O CSV original, o gerador e o notebook 01 conservaram exatamente seus hashes do
início desta sessão. O notebook 01 já tinha alterações anteriores em relação ao
HEAD: elas permanecem e devem ficar fora do staging desta etapa.

## 2. Arquivos criados nesta sessão

- `app/resources.py`: carregamento/cache e preparo automático do demo conhecido.
- `app/metrics.py`: validação estrita das métricas usadas na interface.
- `tests/test_demo_startup.py`: 13 regressões de readiness, cache e corrupção.
- `requirements-dev.txt`: testes, notebooks e ferramentas de navegador.
- `docs/deploy_readiness_plan.md`: escopo e checklist de finalização.
- `reports/deploy_readiness_report.md`: este relatório.
- `reports/dependency_verification.json`: metadados oficiais de nove dependências.
- `reports/demo_startup_verification.json`: primeiro acesso sem artefatos.
- `reports/release_security_verification.json`: varredura e hashes preservados.
- `reports/release_verification.json`: consolidação das execuções finais.
- `docs/screenshots/dashboard-overview.png`.
- `docs/screenshots/dashboard-events.png`.
- `docs/screenshots/dashboard-performance.png`.
- `docs/screenshots/dashboard-mobile.png`.
- `docs/screenshots/dashboard-startup-demo.png`.

Os módulos de calibração/eventos, os demais testes, app e notebooks 02/03 já
existiam no início, embora muitos ainda estejam sem rastreamento no Git.

## 3. Arquivos atualizados nesta sessão

- `app/dashboard.py`: startup seguro, métricas/cache compartilhados, limite de
  quatro análises por 900 segundos, rótulo Forecast e aviso correto sobre score.
- `app/event_views.py`: threshold original explícito e suporte a referência
  histórica nula.
- `requirements.txt`: nove dependências de runtime fixadas e sem ferramentas dev.
- `README.md`: instruções de publicação, demo, resultados, limitações e autor.
- `notebooks/03_anomaly_calibration_and_events.ipynb`: todas as células executadas.
- `reports/model_metrics.json`: atualizado pelo treino completo reproduzível.
- `reports/anomaly_notebook_verification.json`: nova execução sem erros.
- `reports/anomaly_runtime_verification.json`: nova verificação HTTP/navegador.
- `reports/development_report.md`: referência ao workspace convertida para caminho
  relativo; o restante mantém o histórico de desenvolvimento.
- Artefatos ignorados em `models/` e CSVs ignorados em `data/outputs/`: regenerados.

`.gitignore`, configuração Streamlit e outras implementações já possuíam mudanças
anteriores; não foram recriados. Nenhuma alteração foi feita no notebook 01.

## 4. Arquitetura final

```text
CSV → validação → features causais/contextuais
                     ↓
treino temporal → artefatos + política de threshold + métricas
                     ↓
EnergyPipeline.analyze → análise por linha
EnergyPipeline.analyze_with_events → results + events + summary
              ↙                         ↘
     FastAPI/Pydantic             Streamlit/Plotly
                                  ↓ se artefatos inválidos
                          preparo do demo original em cache
```

O pipeline de inferência não chama `fit`. A API exige artefatos persistidos e
retorna 503 quando eles estão indisponíveis. O preparo automático pertence apenas
ao startup do dashboard e não sobrescreve os artefatos configurados pelo usuário.

## 5. Divisão temporal e prevenção de leakage

| Detector | Início | Fim | Linhas | Uso |
|---|---|---|---:|---|
| Train 60% | 01/01/2026 00:00 | 18/01/2026 23:45 | 1.728 | Ajuste, perfil e escala |
| Validation 20% | 19/01/2026 00:00 | 24/01/2026 23:45 | 576 | Seleção de threshold |
| Holdout 20% | 25/01/2026 00:00 | 30/01/2026 23:45 | 576 | Avaliação final |

O detector não é refitado depois da calibração. Holdout não entra na escolha do
threshold. Um teste altera potência e rótulos de todo o holdout e confirma que
threshold, grade, perfil e escala continuam os mesmos. Lags/rolling usam apenas
observações anteriores; rolling aplica shift(1). Forecasting mantém treino 80%,
três janelas internas e holdout 20%, com observações anteriores reais permitidas.

## 6–8. Thresholds e política

- Original, mesmo detector treinado em 60%: **64,664797/100**.
- Calibrado: **68/100**, com decisão `score >= 68`.
- Referência histórica treinada em 80%: **57,299976/100**, em outra escala.
- Grade: 101 thresholds, de 0 a 100, passo 1.
- Política: maximizar F1 na Validation com recall ≥0,80; desempatar por menos FP
  e depois maior threshold. Sem rótulos/classes ou candidato viável, manter corte
  padrão e registrar motivo. Nenhum resultado de holdout altera a política.

**Anomaly score 0–100 não representa probabilidade.** A normalização é congelada
no treino e não depende do lote ou do arquivo enviado.

## 9–12. Métricas e FP/FN antes/depois

Comparação que isola o threshold no mesmo detector treinado em 60%:

| Conjunto / corte | Precision | Recall | F1 | FPR | Specificity | Balanced accuracy | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation original | 0,714286 | 0,833333 | 0,769231 | 0,003509 | 0,996491 | 0,914912 | 5 | 2 | 1 | 568 |
| Validation calibrado | 1,000000 | 0,833333 | 0,909091 | 0 | 1,000000 | 0,916667 | 5 | 0 | 1 | 570 |
| Holdout original | 0,571429 | 0,800000 | 0,666667 | 0,005254 | 0,994746 | 0,897373 | 4 | 3 | 1 | 568 |
| Holdout calibrado | 1,000000 | 0,800000 | 0,888889 | 0 | 1,000000 | 0,900000 | 4 | 0 | 1 | 571 |

Validation: FP **2→0**, FN **1→1**. Holdout: FP **3→0**, FN **1→1**.
Neste holdout houve melhoria observada de F1 sem mudança de recall do mesmo
detector. Isso não garante melhoria em outro período/instalação.

A referência histórica de treino 80% tinha TP=5, FP=4, FN=0, TN=567,
precision=55,6%, recall=100% e F1=71,4%. A mudança completa para treino 60% mais
calibração passa FP 4→0 e FN 0→1. Não se atribui essa perda de recall apenas ao
threshold. Validation tem seis positivos; holdout tem somente cinco.

## 13–16. Eventos operacionais

| Janela | Total | Isolados/suspected | Persistentes/confirmed | Alto | Crítico | Duração total |
|---|---:|---:|---:|---:|---:|---:|
| Dataset completo | 19 | 19 | 0 | 9 | 10 | 0 min |
| Holdout | 4 | 4 | 0 | 1 | 3 | 0 min |

Baixo e Moderado: zero nesse demo. Gap máximo padrão de 15 minutos e mínimo
de dois pontos para confirmar persistência, ambos configuráveis. Pontos normais
interrompem eventos. Severidade agregada é a maior dos pontos.

Duração é fim menos início; um ponto isolado tem zero minutos. O exemplo separado
18:15/18:30/18:45 produz um evento confirmado de 30 minutos. Notebook, testes,
HTTP e upload/download verificaram esse comportamento sem incluir o exemplo nas
métricas do dataset original. Confirmado significa persistência, sem diagnosticar
causa física. Eventos incluem todos os campos pedidos e uma linha por evento no CSV.

## 17–19. Forecast V1, V2 e baselines

| Candidato / holdout | MAE kW | RMSE kW | R² |
|---|---:|---:|---:|
| Forecast V1 | 0,5099 | 1,1495 | 0,8716 |
| Forecast V2 | 0,4806 | 1,0645 | 0,8899 |
| Dia anterior | 0,8707 | 1,7987 | 0,6855 |
| Perfil histórico | **0,4126** | **0,9879** | **0,9051** |

O perfil histórico é o candidato servido, escolhido por menor MAE médio na CV
anterior ao holdout. O baseline supera ML neste demo. Forecast é one-step de
15 minutos, sem alegação de previsão multistep. O fallback V1 continua explícito
quando o candidato selecionado precisa de histórico ausente.

## 20–22. Testes, notebook e treino completo

Comandos executados:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train
```

- **133 testes aprovados**, zero falhas, em 88,25 s na execução final.
- Um aviso de depreciação Starlette/httpx no teste local; nenhuma falha funcional.
- A suíte anterior de 120 testes foi preservada; 13 regressões foram adicionadas.
- Treino completo de 200 árvores concluído com código 0, threshold persistido,
  métricas e cinco CSVs gerados.
- Novo processo carregou os joblibs e analisou 2.880 linhas/19 eventos sem fit.
- Notebook 03: **oito células de código executadas, zero erros**; curva PR,
  fronteiras, thresholds, métricas e exemplos de eventos disponíveis.
- A execução inicial do pytest dentro do sandbox encontrou restrições do Windows
  aos temporários. A suíte final foi executada com acesso local permitido; não se
  alterou código ou reduziu cobertura para contornar essas restrições.
- Revisão independente identificou e confirmou as correções de métricas aninhadas,
  strings numéricas, referência histórica nula e limite do cache. Sem achados pendentes.

Outputs locais: analysis_results.csv (701.615 bytes), anomaly_events.csv (8.191),
forecast_results.csv (51.240), anomaly_points.csv (6.838) e anomaly_thresholds.csv
(10.332). Permanecem ignorados pelo Git.

## 23. API

Servidor iniciado com `python -m uvicorn app.api:app --reload`. Verificação HTTP:

| Endpoint | Resultado válido |
|---|---|
| GET /health | 200, models_ready=true, threshold=68 |
| GET /docs | 200 |
| POST /predict/anomaly | 200 |
| POST /predict/forecast | 200 |
| POST /analyze | 200 |
| POST /analyze/events | 200, resumo e eventos coerentes |

Payloads inválidos retornaram 422. Os testes cobriram 503 em ausência/corrupção
de modelos e mantiveram os schemas anteriores. O endpoint de eventos usa Pydantic.
O lote didático 3+1 pontos retornou dois eventos, um persistente de 30 minutos.

## 24. Streamlit e evidência visual

Servidor iniciado com `python -m streamlit run app/dashboard.py`. Verificados no
Edge headless e por inspeção das screenshots: demo, quatro abas, seis áreas,
cards, gráficos, eventos vazios/persistentes, Validation/Holdout separados,
filtros, upload CSV com/sem ground truth, CSV vazio, coluna ausente, timestamp
inválido e download de eventos. Sem erros JavaScript/console na execução final.
Viewport de 390 px renderizou sem transbordamento horizontal da página.

As três screenshots pedidas estão em `docs/screenshots/dashboard-overview.png`,
`dashboard-events.png` e `dashboard-performance.png`; há capturas adicionais
mobile e do startup sem artefatos. As tabelas largas têm rolagem interna.

## 25. Readiness para Streamlit Community Cloud

Estratégia escolhida: **B**, preparo pequeno e seguro uma vez por processo.
Uma cópia de app/src/raw sem models nem outputs abriu o demo em **24,111 s**;
um upload seguinte levou **4,888 s**, reutilizou os modelos e gerou um evento
persistente. São medições locais, não previsão de latência da nuvem.

O SHA-256 limita o preparo ao demo original. Artefatos temporários são removidos
após carregar os modelos; nada é gravado em models/ do clone. Uploads não treinam.
Recursos locais inválidos ativam aviso explícito de demo. Cache de análises:
quatro entradas, TTL 900 s; cache de conjuntos de recursos: quatro entradas.

Entrypoint: **app/dashboard.py**. Imports independentes do cwd, paths com Path,
sem drive Windows no runtime, sem `.env` ou secrets. Configuração de tema,
upload até 20 MB e telemetria desativada em `.streamlit/config.toml`.

Selecionar **Python 3.13** em Advanced settings. Essa seleção é feita no painel,
conforme a [documentação oficial do Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
Não foi criado runtime.txt, que não substitui a configuração oficial desse serviço.
As dependências seguem a [documentação oficial de dependências](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

PyPI confirmou versões e Requires-Python das nove dependências; a resolução e o
download de **50 wheels**, incluindo dependências transitivas para CPython 3.13
Linux x86_64, passaram. A primeira tentativa usou somente manylinux2014, antiga
demais para as versões numéricas; a resolução correta incluiu manylinux 2.27/2.28
da glibc suportada no ambiente Debian. `pip check` local também passou.

Não se executou o aplicativo em Linux nem se publicou uma URL pública nesta
sessão; essas são as verificações posteriores ao staging/revisão do proprietário.
Os arquivos elegíveis para publicação somam aproximadamente **4,8 MiB**, sem
virtualenv, wheels ou joblibs. Os joblibs locais somam cerca de **10,57 MiB**.

## 26. Dependências finais

| Runtime | Versão |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| plotly | 6.9.0 |
| streamlit | 1.65.0 |
| fastapi | 0.143.0 |
| pydantic | 2.14.0 |
| uvicorn | 0.54.0 |

Desenvolvimento: pytest, httpx, matplotlib, nbformat, nbclient, ipykernel e
Playwright, separados em requirements-dev.txt. Não são necessários ao dashboard.

## 27. Segurança e arquivos fora do commit

Varredura dos arquivos elegíveis não encontrou chaves privadas, padrões conhecidos
de tokens, credenciais atribuídas nem paths pessoais. Não há `.env` elegível.
Isso é uma verificação por padrões, não uma prova absoluta de ausência de segredos.

Não incluir: `.venv/`, `.runtime-tmp/`, `.runtime-tools/`, `.pytest_cache/`,
`__pycache__/`, `models/*.joblib`, `models/metrics.json`, `data/outputs/*.csv`,
logs, downloads temporários, wheels e qualquer `.env`/secret.
**Não incluir inadvertidamente notebooks/01_exploratory_analysis.ipynb.**

Hashes preservados:

```text
notebook 01: ebb1c5c2cba4a09060a0cc34d7234f9588b4f7f6068577ae2054b593b760f258
gerador:    61d95077472f2d91b80eaee9db274bda8207dbdfbb6972babcf03c4fdb2185fe
CSV demo:   fda66ff898f1c8e348fa42cd1b6c1f29a4091f032258f1b3663341e182df0995
```

## 28–29. Limitações e próximos passos

Dados sintéticos de uma instalação, 30 dias, seis positivos na Validation e cinco
no Holdout. Não há evidência de generalização para instalações reais. Todos os
eventos originais são isolados; persistência usa exemplos separados. Não há
ingestão contínua, continuidade de eventos entre chamadas, autenticação, banco ou
diagnóstico físico. Reiniciar/limpar o cache exige novo preparo do demo.

Próximos passos: publicar após revisão/staging seletivo; verificar logs e primeiro
acesso no Cloud; coletar dados reais e mais períodos independentes; validar por
instalação, monitorar drift e acrescentar armazenamento/alertas com revisão humana.
Uma nova decisão de threshold exige nova validação e um novo holdout independente.

## 30. Git e sugestão de commits seletivos

`git status` e `git diff --check` foram verificados ao finalizar. Há mudanças
preexistentes e muitos módulos ainda não rastreados; um diff apenas de arquivos
rastreados não representa toda a implementação. Não houve staging nesta sessão.
Nunca usar `git add .` neste estado.

Sugestão de agrupamento, após inspeção manual de cada arquivo:

1. `feat: finish temporal anomaly calibration and operational events`:
   módulos necessários de src/, API, gráficos, views de eventos, testes anteriores
   da Etapa 2, notebook 03 e métricas de referência.
2. `fix: make Streamlit demo start without local artifacts`:
   app/resources.py, app/metrics.py, app/dashboard.py, app/event_views.py,
   tests/test_demo_startup.py, requirements.txt e requirements-dev.txt.
3. `docs: prepare Energy Intelligence AI for public demo`:
   README, plano/relatório final, evidências JSON e screenshots selecionadas.

Como os diretórios completos ainda incluem trabalho anterior não rastreado,
revise sua inclusão: app/__init__.py, src/__init__.py, pipeline/inferência,
forecasting, validação e configuração também precisam existir no clone público.
Use `git add -- caminho/exato1 caminho/exato2` por grupo. Antes de qualquer commit,
verifique `git diff --cached --stat`, `git diff --cached --check` e confirme que
o notebook 01 e arquivos ignorados não entraram. Commit/push não foram executados.
