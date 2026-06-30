import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

MELHOR_PROFUNDIDADE = 5
MELHOR_K = 9

resultados = []

print("Iniciando as 30 execuções")

for semente in range(30):
    print(f"Executando {semente + 1}/30")

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=semente)

    dt = DecisionTreeClassifier(max_depth=MELHOR_PROFUNDIDADE)
    t0_dt = time.time()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    tempo_dt = time.time() - t0_dt

    resultados.append({
        'Algoritmo': 'Árvore de Decisão',
        'Rodada': semente,
        'Acurácia': accuracy_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt), 
        'F1-Score': f1_score(y_test, y_pred_dt),
        'Tempo (s)': tempo_dt
    })

    knn = KNeighborsClassifier(n_neighbors=MELHOR_K)
    t0_knn = time.time()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    tempo_knn = time.time() - t0_knn

    resultados.append({
        'Algoritmo': 'KNN',
        'Rodada': semente,
        'Acurácia': accuracy_score(y_test, y_pred_knn),
        'Recall': recall_score(y_test, y_pred_knn),
        'F1-Score': f1_score(y_test, y_pred_knn),
        'Tempo (s)': tempo_knn
    })

df_resultados = pd.DataFrame(resultados)

#Estatística Descritiva
print("\n Estatística Descritiva")
estatisticas = df_resultados.groupby('Algoritmo').agg(
    Acurácia_Média=('Acurácia', 'mean'),
    Acurácia_Mediana=('Acurácia', 'median'),
    Acurácia_DesvioPadrao=('Acurácia', 'std'),
    Recall_Médio=('Recall', 'mean'),
    Tempo_Médio=('Tempo (s)', 'mean')
).round(4)
display(estatisticas)

#Teste de Hipótese
print("\n Teste de Hipótese")
acc_arvore = df_resultados[df_resultados['Algoritmo'] == 'Árvore de Decisão']['Acurácia']
acc_knn = df_resultados[df_resultados['Algoritmo'] == 'KNN']['Acurácia']


t_stat, p_value = stats.ttest_rel(acc_arvore, acc_knn)
print(f"Valor-P (p-value) do teste: {p_value:.5f}")
if p_value < 0.05:
    print("Conclusão: A diferença entre os modelos é significativa")
else:
    print("Conclusão: não há diferença estatística entre os modelos.")

#Boxplots
plt.figure(figsize=(12, 5))

# Boxplot de Acurácia
plt.subplot(1, 2, 1)
sns.boxplot(x='Algoritmo', y='Acurácia', data=df_resultados, palette='Set2')
plt.title('Comparação de Acurácia (30 Execuções)')

# Boxplot de Tempo
plt.subplot(1, 2, 2)
sns.boxplot(x='Algoritmo', y='Tempo (s)', data=df_resultados, palette='Set1')
plt.title('Comparação de Tempo de Execução')

plt.tight_layout()
plt.show()
