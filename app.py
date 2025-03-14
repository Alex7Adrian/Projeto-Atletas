from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)


df = pd.read_csv('dados_atletas.csv')
X = df[['velocidade', 'força', 'resistencia']]

# Aplicar aprendizado não supervisionado (K-Means)
num_grupos = 3
kmeans = KMeans(n_clusters=num_grupos, random_state=42, n_init=10)
df['grupo_kmeans'] = kmeans.fit_predict(X)

# Criar modelo supervisionado (Regressão Linear)
modelo_regressao = LinearRegression()
modelo_regressao.fit(X, df['grupo_kmeans'])
y_pred = modelo_regressao.predict(X)

# Coeficientes da Regressão Linear
coeficientes = modelo_regressao.coef_
intercepto = modelo_regressao.intercept_


if not os.path.exists("static"):
    os.makedirs("static")

# Gerar gráfico da regressão linear
plt.figure(figsize=(6, 4))
plt.scatter(df.index, df['grupo_kmeans'], label="Grupos reais", color='blue')
plt.plot(df.index, y_pred, label="Regressão Linear", color='red')
plt.xlabel("Atletas")
plt.ylabel("Grupo")
plt.legend()
plt.title("Regressão Linear (Supervisionado)")
plt.savefig("static/grafico_regressao.png")
plt.close()

# Gerar gráfico do K-Means (Aprendizado Não Supervisionado)
plt.figure(figsize=(6, 4))
plt.scatter(df['velocidade'], df['resistencia'], c=df['grupo_kmeans'], cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroides')
plt.xlabel("Velocidade")
plt.ylabel("Resistência")
plt.legend()
plt.title("K-Means (Não Supervisionado)")
plt.savefig("static/grafico_kmeans.png")
plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    erro_mse = None
    resultado_regressao = None
    grupo_kmeans = None
    valores_usuario = None
    formula_regressao = None

    if request.method == "POST":
        # Receber dados do formulário
        velocidade = float(request.form["velocidade"])
        forca = float(request.form["forca"])
        resistencia = float(request.form["resistencia"])

        # Criar DataFrame do novo atleta
        novo_atleta = pd.DataFrame([[velocidade, forca, resistencia]], columns=['velocidade', 'força', 'resistencia'])

        # Salvar o novo atleta no arquivo CSV
        df = pd.read_csv('dados_atletas.csv')
        df = pd.concat([df, novo_atleta], ignore_index=True)
        df.to_csv('dados_atletas.csv', index=False)

        # Re-treinar o modelo K-Means com os novos dados
        X = df[['velocidade', 'força', 'resistencia']]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['grupo_kmeans'] = kmeans.fit_predict(X)

        # Re-treinar o modelo de Regressão Linear
        modelo_regressao = LinearRegression()
        modelo_regressao.fit(X, df['grupo_kmeans'])
        y_pred = modelo_regressao.predict(X)

        # Coeficientes da Regressão Linear
        coeficientes = modelo_regressao.coef_
        intercepto = modelo_regressao.intercept_

        # Calcular erro da Regressão Linear
        erro_mse = mean_squared_error(df['grupo_kmeans'], y_pred)

        # Realizar a previsão do grupo no K-Means para o atleta inserido
        entrada_usuario = np.array([[velocidade, forca, resistencia]])
        grupo_kmeans = kmeans.predict(entrada_usuario)[0]  # Previsão do grupo do atleta

        # Gerar gráficos novamente com os novos dados
        if not os.path.exists("static"):
            os.makedirs("static")

        # Gerar gráfico da regressão linear
        plt.figure(figsize=(6, 4))
        plt.scatter(df.index, df['grupo_kmeans'], label="Grupos reais", color='blue')
        plt.plot(df.index, y_pred, label="Regressão Linear", color='red')
        plt.xlabel("Atletas")
        plt.ylabel("Grupo")
        plt.legend()
        plt.title("Regressão Linear (Supervisionado)")
        plt.savefig("static/grafico_regressao.png")
        plt.close()

        # Gerar gráfico do K-Means (Aprendizado Não Supervisionado)
        plt.figure(figsize=(6, 4))
        plt.scatter(df['velocidade'], df['resistencia'], c=df['grupo_kmeans'], cmap='viridis', alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroides')
        plt.scatter(velocidade, resistencia, color='orange', s=100, label="Atleta Inserido", edgecolor='black', linewidth=2)
        plt.xlabel("Velocidade")
        plt.ylabel("Resistência")
        plt.legend()
        plt.title("K-Means (Não Supervisionado) com Destaque do Atleta")
        plt.savefig("static/grafico_kmeans_com_usuario.png")
        plt.close()

        # Guardar os valores inseridos pelo usuário
        valores_usuario = {
            "velocidade": velocidade,
            "forca": forca,
            "resistencia": resistencia
        }

        # Fórmula usada na regressão linear
        formula_regressao = f"Grupo = ({coeficientes[0]:.2f} * Velocidade) + ({coeficientes[1]:.2f} * Força) + ({coeficientes[2]:.2f} * Resistência) + ({intercepto:.2f})"

    return render_template("index.html",
                           erro_mse=erro_mse,
                           grupo_kmeans=grupo_kmeans,
                           valores_usuario=valores_usuario,
                           formula_regressao=formula_regressao)


if __name__ == "__main__":
    app.run(debug=True, threaded=False)

