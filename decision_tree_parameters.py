import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

#profundidades 
profundidades = [2, 3, 5, 7, 10, 15, 20]

acuracias = []
tempos = []

print("Testes com a Árvore de Decisão\n")

for prof in profundidades:
    inicio = time.time()

    arvore = DecisionTreeClassifier(max_depth=prof, random_state=42)
    arvore.fit(x_treino, y_treino)

    previsoes = arvore.predict(x_teste)

    fim = time.time()
    acuracia = accuracy_score(y_teste, previsoes)

    acuracias.append(acuracia)
    tempos.append(fim - inicio)

    print(f"Profundidade {prof:2d} | Acurácia: {acuracia*100:.2f}% | Tempo: {fim-inicio:.3f}s")

plt.figure(figsize=(8, 5))
plt.plot(profundidades, acuracias, marker='o', linestyle='-', color='blue')
plt.title('Influência da Profundidade na Acurácia')
plt.xlabel('Profundidade Máxima')
plt.ylabel('Acurácia (0 a 1)')
plt.grid(True)
plt.show()
