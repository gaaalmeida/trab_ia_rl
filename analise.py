import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv('consumo_cerveja_sp.csv', sep=',')

dados.head()
dados.isna().sum()
dados.describe()

# Removendo NaN
dadosc = dados.copy()
dadosc.dropna(inplace=True)
print(dadosc.isna().sum())

dadosc.describe().round(2)

# Heatmap de correlação
sns.heatmap(dadosc.corr().round(4), annot=True)
plt.show()

# Outliers
out = sns.boxplot(data=dadosc['consumo'], width=0.6, orient='v')

out.figure.set_size_inches(9,5)
out.set_title('Consumo em 1 ano', fontsize=16)
out.set_ylabel('Consumo em l(litros)', fontsize=10)

plt.show()

# Representando a variável Y
f, ry = plt.subplots(figsize=(16,3))

ry.set_title('Consumo ao longo de 1 ano', fontsize=18)
ry.set_ylabel('Consumo em l(litros)', fontsize=13)
ry.set_xlabel('Tempo (dias)', fontsize=13)

ry = dadosc['consumo'].plot(fontsize=16)

plt.show()

# Relação entre consumo na semana e no final de semana
rcnfds = sns.boxplot(data=dadosc, y='consumo', x='finaldesemana', orient='v', width=0.6)

rcnfds.figure.set_size_inches(9,5)
rcnfds.set_title('Consumo da semana em relação ao final de semana')
rcnfds.set_ylabel('Consumo em l(litros)', fontsize=13)
rcnfds.set_xlabel('Final de semana (0 = Não, 1 = Sim)', fontsize=13)

plt.show()

# Dispersão das variáveis

dis = sns.pairplot(
    dadosc, y_vars='consumo', x_vars=['precipitacao', 'finaldesemana', 'temp_min', 'temp_med', 'temp_max'],
    height=4,
    kind='reg', plot_kws={'line_kws':{'color':'red'}}
    )
dis.fig.suptitle('Dispersão entre as variáveis', y=1.2, fontsize=22)
plt.show()