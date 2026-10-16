# Etapa 2 — Calibração e eventos operacionais

**Objetivo:** calibrar o threshold exclusivamente na validação temporal e
agrupar pontos detectados sem quebrar a análise por linha, API ou forecasting.

## Auditoria e desenho

O detector atual usa `model.predict()`, contaminação fixa 20/2880 e perfil horário
congelado. O score normaliza `-decision_function` com mínimos/máximos do treino;
severidade considera a detecção antes dos thresholds 40/60/80. Métricas usam
somente rótulos explícitos. O pipeline persiste os artefatos e não faz fit ao inferir.

Preservar essa arquitetura; adicionar módulos de calibração e eventos em `src/`.
Calibrar sobre o mesmo conjunto usado no fit invalidaria a avaliação. Usar holdout
para decidir melhorias também invalidaria o teste final. A escolha adotada é
treino 60%, validação 20%, holdout 20% para anomalias, com fronteiras registradas.
Forecasting mantém seu protocolo existente de treino 80%, CV interna e holdout 20%.
O perfil/score do detector são congelados nos primeiros 60%; não refitar o detector
depois de calibrar, pois isso mudaria a escala do threshold.

Comparar três referências: V2 original treinado em 80% (reprodução do estado
anterior), detector treinado em 60% com corte padrão e o mesmo detector 60% com
corte calibrado. As duas últimas isolam o efeito do threshold. Nenhuma referência
de holdout participa da seleção.

Política: score >= threshold; grade 0–100 em passos de 1; maximizar F1 mantendo
recall >= 0,80; desempatar por menos FP e depois maior threshold. Sem rótulos,
sem ambas as classes ou sem candidato viável, conservar o corte padrão e registrar
o motivo. Contaminação permanece fixa e deixa de ser o único corte decisório.

Eventos agrupam pontos detectados adjacentes, com gap máximo configurável de 15
minutos. Uma medição normal interrompe o evento mesmo com tolerância maior.
Pontos isolados permanecem `suspected`; >=2 pontos são `confirmed`, indicando
persistência, sem confirmar causa física. Duração é fim menos início: um ponto
tem 0 minutos. Severidade é a maior observada. A potência esperada é a média do
forecast; desvio máximo é o desvio de maior magnitude, com sinal preservado.
IDs derivam do início do evento para estabilidade nas mesmas medições.

## Execução e arquivos

- [x] Auditar módulos, testes, API/dashboard e registrar hashes iniciais.
- [x] Escrever testes red de calibração, guardas temporais e métricas adicionais.
  Exemplo: scores [10,50,70,90], labels [0,0,1,1], grade [0,50,60,80]
  selecionam 60 (F1=1, FP=0, recall=1), sem usar holdout.
- [x] Implementar `src/config.py`, `src/anomaly_calibration.py` e ampliar
  `src/anomaly_detection.py`. Verificar casos com rótulos parciais/ausentes,
  classes insuficientes e candidatos sem recall mínimo.
- [x] Escrever testes red de agrupamento/severidade/mensagens/zero eventos.
  Três pontos de 15 minutos devem gerar um evento confirmado de 30 minutos.
- [x] Implementar `src/events.py` e agregação em `src/severity.py`; verificar
  interrupções por normalidade, lacunas, tolerância, referência zero e IDs.
- [x] Integrar `src/train.py` e `src/inference.py`; testar mutação de todo o
  holdout sem mudança em threshold/score/perfil e persistência sem fit.
  Salvar `anomaly_thresholds.csv`, pontos em `anomaly_points.csv` e eventos
  agregados em `anomaly_events.csv`; conservar análise/forecast existentes.
- [x] Adicionar `POST /analyze/events` com Pydantic e testes 200/422/503.
  Preservar schemas e respostas dos endpoints atuais.
- [x] Evoluir `app/dashboard.py`, `app/charts.py`; aba Eventos com quatro cards,
  tabela/timeline/download, comparação Validation/Holdout na Performance.
  Corrigir identidade do seletor de datas ao trocar CSV/janela e verificar
  estado vazio/com eventos via AppTest e navegador real.
- [x] Executar suíte completa e treino de 200 árvores; registrar métricas reais.
- [x] Criar e executar `notebooks/03_anomaly_calibration_and_events.ipynb`, com
  grade, seleção, curva PR de validação, holdout e exemplos de eventos.
- [x] Revisão independente conforme skill de revisão de código; corrigir achados.
- [x] Atualizar README, screenshot de Eventos e relatório com resultados/limitações.
- [x] Verificar preservação de notebook 01, gerador e CSV; não fazer commit/push.

Comandos: `.\.venv\Scripts\python.exe -m pytest -q` e
`.\.venv\Scripts\python.exe -m src.train`. Verificações do ambiente Windows
usam execução fora do sandbox quando o paralelismo exige pipes bloqueados.
