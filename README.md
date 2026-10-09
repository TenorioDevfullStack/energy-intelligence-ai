# ⚡ Energy Intelligence AI

Plataforma de análise inteligente de dados elétricos utilizando Data Science e Machine Learning.

## 🎯 Objetivo

O Energy Intelligence AI tem como objetivo analisar o comportamento energético de uma instalação elétrica, identificar padrões de consumo, detectar anomalias e evoluir para previsão de demanda, consumo e geração automática de insights.

## 📊 Funcionalidades implementadas

- Geração de dataset elétrico sintético
- Análise exploratória dos dados
- Curva de carga de 30 dias
- Perfil médio de carga por horário
- Comparação entre dias úteis e finais de semana
- Análise de consumo por período do dia
- Detecção de anomalias com Isolation Forest
- Feature engineering com contexto temporal
- Validação do modelo com ground truth

## 🤖 Machine Learning

Foi utilizado o algoritmo não supervisionado `Isolation Forest` para detectar comportamentos anormais no consumo elétrico.

### Modelo V1

Utilizando apenas variáveis elétricas:

- 11 de 20 anomalias detectadas
- Recall: 55%

### Modelo V2

Após feature engineering baseado no comportamento horário da instalação:

- 16 de 20 anomalias detectadas
- Precision: 80%
- Recall: 80%
- F1-score: 80%

A melhoria mostra a importância do contexto temporal para detecção de anomalias em sistemas de energia.

## 🧠 Features utilizadas

- Tensão
- Corrente
- Potência ativa
- Fator de potência
- Energia consumida
- Potência mediana por horário
- Desvio da potência em relação ao padrão
- Razão entre potência atual e potência típica

## 🛠️ Tecnologias

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 📁 Estrutura do projeto

```text
energy-intelligence-ai/
├── data/
│   └── raw/
├── notebooks/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```
