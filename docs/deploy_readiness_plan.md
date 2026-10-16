# Finalização da Etapa 2 e preparação para publicação

Escopo autorizado: preservar a arquitetura, os trabalhos existentes e o notebook
01; concluir validação, documentação e inicialização pública. Sem commit/push.

## Decisão de inicialização

Estratégia B: carregar os artefatos locais quando válidos; na ausência/corrupção,
preparar o mesmo pequeno demo (2.880 pontos, 200 árvores, semente 42) uma vez por
processo com `st.cache_resource`. Somente o CSV sintético conhecido pode alimentar
esse preparo. Uploads nunca treinam modelos. Artefatos temporários são removidos
depois do carregamento em memória; a API continua exigindo treino explícito.

## Verificações

- [x] Auditar calibração, eventos, features causais, inferência e APIs existentes.
- [x] Executar treino completo e conferir métricas/outputs.
- [x] Criar regressões de startup sem modelos/com corrupção, cache sem fit e CSV.
- [x] Implementar recursos de demo e integrar sem alterar o contrato do pipeline.
- [x] Separar dependências de runtime/desenvolvimento e documentar Python 3.13
      selecionado em Advanced settings do Streamlit Cloud.
- [x] Validar notebook 03, suíte completa, API HTTP e navegação/upload/download.
- [x] Atualizar screenshots, README e relatório final, com métricas reais.
- [x] Revisar segurança, arquivos publicáveis, hashes preservados e diff --check.

Arquivos de implementação: `app/resources.py`, `app/dashboard.py`,
`app/event_views.py`, `tests/test_demo_startup.py`, `requirements*.txt`.
As áreas de curva de carga e forecast permanecem na Visão geral, com rótulos
explícitos. Validation e Holdout permanecem separados na Performance.
