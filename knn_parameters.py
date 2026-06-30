import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

#valores de K 
valores_k = [1, 3, 5, 7, 9, 11, 15]

acuracias_knn = []
tempos_knn = []

print("Testes com o KNN\n")

for k in valores_k:
    inicio = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_treino, y_treino)

    previsoes = knn.predict(x_teste)

    fim = time.time()
    acuracia = accuracy_score(y_teste, previsoes)

    acuracias_knn.append(acuracia)
    tempos_knn.append(fim - inicio)

    print(f"Vizinhos (K) {k:2d} | Acurácia: {acuracia*100:.2f}% | Tempo: {fim-inicio:.3f}s")

plt.figure(figsize=(8, 5))
plt.plot(valores_k, acuracias_knn, marker='o', linestyle='-', color='green')
plt.title('Influência do Parâmetro K na Acurácia')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia (0 a 1)')
plt.grid(True)
plt.show()
