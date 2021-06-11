import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)

print("Teste: ", X_test.shape)
print("Treino: ", X_train.shape)

m = LinearRegression()
m.fit(X_train, y_train)
print(f"\nTreino: \nQuanto mais proximo de 1, mais preciso é o valor estimado!\nR^2 = {m.score(X_train, y_train).round(2)}\n")

# Estimando
y_estimado = m.predict(X_test)

print(f"\nTestes:\nQuanto mais proximo de 100, mais preciso foi o valor estimado!\nPrecisão de: {metrics.r2_score(y_test, y_estimado).round(2) * 100}%\n")
i = m.intercept_
c = m.coef_

# Resultado
print(f"Se chover há uma redução media de {round(c[1],3)} litros de cerveja.")
print(f"Nos finais de semana o consumo de cerveja sobe em média {round(c[2], 3)} litros.")