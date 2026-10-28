# 📊 Classificador de Risco de Cliente — Consultoria de Investimentos

Modelo de **Machine Learning** que classifica clientes em **baixo, médio ou alto risco financeiro**, etapa obrigatória antes de qualquer recomendação de investimento (suitability — regulação CVM). O modelo é servido por uma **API Flask** com **testes automatizados** e uma **interface web**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Silvio-DON/crc-investimentos/blob/master/notebook/classificador_risco_cliente.ipynb)

## 🎯 Resultado

| Item | Detalhe |
|---|---|
| Modelos comparados | KNN · Árvore de Decisão · Naive Bayes · SVM |
| Otimização | `GridSearchCV` + `StratifiedKFold` (k=5), tudo dentro de `Pipeline` (sem data leakage) |
| Melhor modelo | **SVM (kernel RBF)** — **79% de acurácia e F1** no conjunto de teste |
| Entrega | Pipeline exportado com `joblib` e servido via API REST |

## 🏗️ Arquitetura

```
notebook/   → geração do dataset, análise exploratória, treino e comparação dos modelos
backend/    → API Flask (/health, /predict) + modelo .pkl + testes (pytest)
frontend/   → página HTML que consome a API
```

```
[Formulário web] → POST /predict (JSON) → [Flask + Pipeline scikit-learn] → risco + probabilidades
```

## 🚀 Como rodar

```bash
pip install -r backend/requirements.txt
python notebook/gerar_dataset.py      # gera clientes.csv
python notebook/treinador_modelo.py   # treina e salva backend/model/model.pkl
python backend/app.py                 # sobe a API em http://localhost:5000
pytest backend/tests/test_model.py -v # testa métricas mínimas (≥ 75%)
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "idade": 35, "renda_mensal": 9000, "patrimonio": 150000, "historico_credito": 3,
  "tempo_emprego": 8, "dividas": 5000, "dependentes": 1, "tem_imovel": 1,
  "tem_investimentos": 1, "score_credito": 780
}'
```

Resposta: `{"resultado": "Baixo Risco", "confianca": ..., "probabilidades": {...}}`

## 🔒 Segurança e LGPD

O notebook traz uma seção sobre desenvolvimento seguro: anonimização/pseudonimização de dados financeiros sensíveis, LGPD (Lei 13.709/2018) e normas da CVM.

## ⚠️ Limitações

- O dataset é **sintético** (1.000 clientes), gerado com regras inspiradas no mercado financeiro brasileiro. Em produção, o modelo precisaria ser validado com dados reais.

## 🧰 Stack

Python · pandas · scikit-learn · Flask · pytest · joblib · HTML/JS · Google Colab

---
Feito por [Silvio Dall'Oglio Neto](https://github.com/Silvio-DON) — Pós-graduação em Engenharia de Software, PUC-Rio.
