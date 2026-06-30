from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

x = dados.drop('Diabetes_binary', axis=1)
y = dados['Diabetes_binary']

df_combined = pd.concat([x, y], axis=1)
df_combined = df_combined.dropna()
x = df_combined.drop('Diabetes_binary', axis=1)
y = df_combined['Diabetes_binary']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Dados separados {len(x_treino)} para treino e {len(x_teste)} para teste.')

print("\n Treinando KNN(k=5)")
inicio_knn = time.time()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_treino, y_treino)
fim_knn = time.time()
previsoes_knn = knn.predict(x_teste)
acuracia_knn = accuracy_score(y_teste, previsoes_knn)
tempo_knn = fim_knn - inicio_knn

print(f'Acurácia KNN: {acuracia_knn* 100:.2f}%')
print(f'Tempo de Treinamento KNN: {tempo_knn:.3f} segundos')
