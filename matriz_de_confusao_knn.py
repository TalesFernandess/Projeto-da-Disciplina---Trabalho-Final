import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

melhor_knn = KNeighborsClassifier(n_neighbors=5)
melhor_knn.fit(x_treino, y_treino)
previsoes_knn = melhor_knn.predict(x_teste)

matriz = confusion_matrix(y_teste, previsoes_knn)

plt.figure(figsize=(7, 5.5), dpi=150)
ax = sns.heatmap(matriz, annot=True, fmt='d', cmap='Greens', linewidths=1, linecolor='white', annot_kws={"size": 13, "weight": "bold"}, cbar=False)

ax.set_xticklabels(['Saudável (0)', 'Diabético (1)'], fontsize=11)
ax.set_yticklabels(['Saudável (0)', 'Diabético (1)'], fontsize=11, rotation=0)

plt.title('Matriz de Confusão - KNN', fontsize=14, pad=15, weight='bold')
plt.xlabel('Previsão do Modelo', fontsize=12, labelpad=10, weight='bold')
plt.ylabel('Diagnóstico Real', fontsize=12, labelpad=10, weight='bold')

plt.tight_layout()
plt.show()

relatorio_knn = classification_report(y_teste, previsoes_knn, target_names=['Saudável (0)', 'Diabético (1)'])

print("\nRelatório de Métricas do KNN")
print(relatorio_knn)
