import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

idade = np.random.randint(18, 75, n)
renda = np.random.randint(1500, 30000, n)
patrimonio = np.random.randint(0, 500000, n)
historico_credito = np.random.randint(0, 4, n)
tempo_emprego = np.random.randint(0, 40, n)
dividas = np.random.randint(0, 50000, n)
dependentes = np.random.randint(0, 6, n)
tem_imovel = np.random.randint(0, 2, n)
tem_investimentos = np.random.randint(0, 2, n)
score = np.random.randint(300, 1000, n)

risco = []
for i in range(n):
    pontos = 0
    if historico_credito[i] >= 2: pontos += 2
    if renda[i] > 8000: pontos += 2
    if patrimonio[i] > 100000: pontos += 2
    if dividas[i] < 10000: pontos += 1
    if tempo_emprego[i] > 5: pontos += 1
    if score[i] > 700: pontos += 2
    if tem_imovel[i] == 1: pontos += 1
    if tem_investimentos[i] == 1: pontos += 1
    if dependentes[i] <= 2: pontos += 1
    if idade[i] >= 30 and idade[i] <= 60: pontos += 1

    if pontos <= 6:
        risco.append(2)   # Alto risco
    elif pontos <= 9:
        risco.append(1)   # Médio risco
    else:
        risco.append(0)   # Baixo risco

df = pd.DataFrame({
    'idade': idade,
    'renda_mensal': renda,
    'patrimonio': patrimonio,
    'historico_credito': historico_credito,
    'tempo_emprego': tempo_emprego,
    'dividas': dividas,
    'dependentes': dependentes,
    'tem_imovel': tem_imovel,
    'tem_investimentos': tem_investimentos,
    'score_credito': score,
    'risco': risco
})

df.to_csv('notebook/clientes.csv', index=False)
print(f'Dataset gerado! {len(df)} registros.')
print(df['risco'].value_counts().rename({0:'Baixo', 1:'Médio', 2:'Alto'}))