import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv('diabetes.csv')

linhas, colunas = dados.shape
print('Quantidade de pacientes:', linhas)
print('Quantidade de colunas:', colunas)

print("\n Primeiras linhas da tabela:")
print(dados.head())

contagem_diabetes = dados['Diabetes_binary'].value_counts()

plt.bar(['Sem Diabetes (0)', 'Com Diabetes (1)'], contagem_diabetes)
plt.title('Distribuição de Pacientes - ODS 3')
plt.xlabel('Diagnóstico')
plt.ylabel('Quantidade de Pessoas')
plt.show()
