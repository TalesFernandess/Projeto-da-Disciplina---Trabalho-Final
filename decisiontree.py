from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

x = dados.drop('Diabetes_binary', axis=1)
y = dados['Diabetes_binary']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Dados separados {len(x_treino)} para treino e {len(x_teste)} para teste.')

print("\n Árvore de Decisão(Profundidade=5)")
inicio_arvore = time.time()

arvore = DecisionTreeClassifier(max_depth=5)
arvore.fit(x_treino, y_treino)
fim_arvore = time.time()
previsoes_arvore = arvore.predict(x_teste)
acuracia_arvore = accuracy_score(y_teste, previsoes_arvore)
tempo_arvore = fim_arvore - inicio_arvore

print(f'Acurácia Árvore: {acuracia_arvore* 100:.2f}%')
print(f'Tempo de execução: {tempo_arvore:.3f} segundos')
