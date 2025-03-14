import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Carregar os dados
df = pd.read_csv('dados_atletas.csv')

# Definir variáveis de entrada (X) e saída (y)
X = df[['velocidade', 'força', 'resistencia']]
y = df['grupos']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)


# Criar gráfico da relação entre velocidade e grupo previsto
plt.figure(figsize=(8, 5))
plt.scatter(X_test['velocidade'], y_test, color='blue', label="Grupos reais")
plt.scatter(X_test['velocidade'], y_pred, color='red', label="Previsões")
plt.plot(X_test['velocidade'], y_pred, color='green', linewidth=2, label="Regressão Linear")
plt.xlabel("Velocidade")
plt.ylabel("Grupo")
plt.legend()
plt.title("Regressão Linear - Previsão de Grupos com Base na Velocidade")
plt.savefig("static/grafico.png")
plt.show()

# Função para prever o grupo de um novo atleta
def prever_grupo(velocidade, forca, resistencia):
    novo_atleta = pd.DataFrame([[velocidade, forca, resistencia]], columns=['velocidade', 'força', 'resistencia'])
    grupo_previsto = modelo.predict(novo_atleta)
    return round(grupo_previsto[0])
