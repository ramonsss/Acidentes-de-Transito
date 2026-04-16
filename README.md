# 🚗 Análise de Acidentes Rodoviários (DATATRAN 2021)

Este projeto é um dashboard interativo desenvolvido em **Python + Streamlit**, com foco em **análise exploratória, estatística inferencial e modelagem preditiva** de acidentes rodoviários no Brasil.

Os dados utilizados são do **DATATRAN 2021**, contendo informações detalhadas sobre acidentes em rodovias federais.

---

## 🎯 Objetivo

O objetivo do projeto é transformar dados brutos de acidentes em **insights estatísticos e visuais**, permitindo entender:

- Quando os acidentes acontecem com mais frequência
- Quais fatores estão associados à gravidade
- Se existem diferenças estatisticamente significativas entre grupos
- Como variáveis influenciam a severidade dos acidentes

---

## 🧹 Limpeza e Preparação dos Dados

Antes da análise, foi realizada a preparação do dataset:

- Remoção de espaços em nomes de colunas
- Conversão de variáveis de tempo (`horario → hora`)
- Tratamento de valores nulos (`dropna`, `errors="coerce"`)
- Padronização de variáveis categóricas
- Criação de variáveis derivadas como **severidade**

---

## ⚠️ Criação da Variável de Severidade

Como o dataset não possui uma métrica direta de gravidade, foi criada uma variável composta:

```python
severidade = mortos * 10 + feridos_leves * 2 + pessoas * 0.5