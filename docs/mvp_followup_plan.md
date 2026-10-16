# Continuação do MVP — 13/10/2026

**Objetivo:** concluir a auditoria e verificar o MVP existente, corrigindo lacunas
de inferência e interface sem substituir os modelos, notebooks ou resultados.

**Arquitetura:** preservar os módulos em `src/`, o pipeline persistido e os
consumidores independentes FastAPI e Streamlit. A disponibilidade da previsão
deve depender das features do candidato selecionado. Os filtros de data devem
acompanhar a fonte de dados e a janela elegível.

**Stack:** Python, pandas, scikit-learn, joblib, FastAPI/Pydantic,
Streamlit/Plotly, pytest, AppTest e Playwright para verificação local.

## Estado inicial e decisões

- [x] Ler scripts, notebooks, testes, configuração, README e relatórios existentes.
- [x] Registrar hashes iniciais e alterações Git preexistentes.
- [x] Executar suíte inicial: 53 testes aprovados; um aviso de depreciação do
  TestClient de Starlette, sem falhas.

Todos os subsistemas principais já existem: lags/rolling causais, três janelas
TimeSeriesSplit, holdout final de 20%, baselines, detector contextual, score,
severidade, métricas, persistência, API, upload, gráficos e documentação.

A alternativa de reconstruir componentes duplicaria o trabalho existente.
Adicionar novos subsistemas (banco, alertas, autenticação) ampliaria o MVP pedido.
A abordagem adotada é corrigir os contratos atuais e verificá-los de ponta a ponta.
O usuário autorizou execução autônoma no workspace e pediu somente sugestões de
commits: manter o checkout atual, preservar alterações locais e não fazer commits
ou push nesta rodada.

## 1. Inferência por candidato

Arquivos: `src/forecasting.py`, `tests/test_models.py`, `tests/test_api.py`.

- [ ] Escrever testes para V1 e perfil sem histórico de sete dias, baseline diário
  com lag de 24 horas disponível/ausente, e V2 com histórico completo/incompleto.
  Verificar também consistência entre análise individual e em lote para o perfil.

Exemplo de contrato a verificar:

```python
bundle = dict(pipeline.forecast, selected="baseline_profile")
single = predict_forecast(demo.tail(1).reset_index(drop=True), bundle)
assert single.forecast_source.iloc[0] == "baseline_profile"
```

- [ ] Executar os novos testes antes de mudar a implementação e confirmar falha
  no requisito de histórico indevido.
- [ ] Selecionar o caminho pelo candidato: V1 usa tempo; perfil usa as medianas
  congeladas; diário requer somente `lag_96`; V2 requer todas as suas features.
  Usar `forecast_v1_fallback` somente quando faltarem features do diário/V2.
- [ ] Ajustar expectativas antigas de fallback e verificar os módulos ML/API.

## 2. Filtros do dashboard

Arquivos: `app/dashboard.py`, `tests/test_dashboard.py`.

- [ ] Reproduzir via AppTest a troca de dataset/período após selecionar datas.
  Ao mudar o filtro após treino, o período deve cobrir a nova janela elegível.
- [ ] Confirmar falha antes da correção.
- [ ] Calcular o contexto da janela por conteúdo CSV, filtro e versões dos modelos;
  limpar somente o estado do widget `period` quando esse contexto mudar.
- [ ] Atualizar o texto do fallback para distinguir os requisitos diário/V2.
- [ ] Verificar dataset, filtros, seis gráficos, tabelas e exportações.

## 3. Reprodutibilidade e execução real

- [ ] Executar `.\.venv\Scripts\python.exe -m src.train` (200 árvores), reproduzir
  métricas e gerar modelos/CSVs. Verificar a inferência contra os CSVs de holdout.
- [ ] Executar suíte completa e `pip check`; investigar falhas reais.
- [ ] Executar notebook 02 em uma cópia temporária, preservando ambos os notebooks.
- [ ] Reiniciar API/dashboard e verificar HTTP 200/422, modelos prontos e forecast
  do próximo intervalo, além da interface no Edge/Playwright com upload válido,
  CSV inválido, troca de dataset, filtros e downloads.
- [ ] Solicitar revisão independente conforme skill de revisão de código; resolver
  achados relevantes e repetir apenas as verificações afetadas.

## 4. Documentação e entrega

Arquivos: `README.md`, `reports/mvp_followup_report.md` e evidências de verificação.

- [ ] Documentar requisitos de histórico por candidato e resultado da auditoria.
- [ ] Registrar arquivos realmente criados/alterados nesta rodada, métricas,
  testes, arquitetura, comandos, limitações e sugestões de commits separados.
- [ ] Comparar hashes do notebook original, notebook 02, gerador e CSV; verificar
  `git diff --check`. Não atribuir alterações Git preexistentes a esta rodada.

Comandos de verificação (a partir da raiz):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.port 8501
```

Neste ambiente, os recursos de paralelismo dos modelos exigem executar os processos
fora do sandbox restrito (que retorna WinError 5 ao criar pipes do Windows).
