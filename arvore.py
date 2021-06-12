import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score
from sklearn import metrics

dados = pd.read_csv('consumo_cerveja_sp.csv', sep=',')

dados.head()
dados.isna().sum()
dados.describe()

# Removendo NaN
dadosc = dados.copy()
dadosc.dropna(inplace=True)
print(dadosc.isna().sum())

dadosc.describe().round(2)
print()

# Regressão linear
y = dadosc['consumo']
X = dados[['temp_max', 'precipitacao', 'finaldesemana']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
arvore = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
arvore = arvore.fit(X_train, y_train)
y_estimado = arvore.predict(X_train)
print(f"Acuracia: {accuracy_score(y_train, y_estimado)}")
print(f"Precisao: {precision_score(y_train, y_estimado, average='weighted', zero_division=0)}")


tree.plot_tree(arvore, fontsize=8)
plt.show()
