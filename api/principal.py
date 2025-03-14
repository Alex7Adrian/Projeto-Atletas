import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Gerar dataset
np.random.seed(42)
num_atletas = 100

dados = {
    'velocidade': np.random.uniform(5, 25, num_atletas),
    'força': np.random.uniform(50, 200, num_atletas),
    'resistencia': np.random.uniform(30, 120, num_atletas)
}

df = pd.DataFrame(dados)

# Normalizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Testar diferentes números de clusters usando a métrica da Inércia
inercia = []
valores_k = range(2, 6)

for k in valores_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inercia.append(kmeans.inertia_)

# Escolher o melhor número de clusters com base na curvatura (cotovelo)
melhor_k = valores_k[inercia.index(min(inercia))]

# Aplicar K-Means com o melhor K
kmeans = KMeans(n_clusters=melhor_k, random_state=42)
df['grupos'] = kmeans.fit_predict(df_scaled)

# Salvar os dados processados
df.to_csv('dados_atletas.csv', index=False)
print(f"Dados dos atletas salvos em 'dados_atletas.csv' com {melhor_k} grupos")

# Exibir gráfico do método do cotovelo
plt.plot(valores_k, inercia, marker='o', linestyle='--')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Seleção de K')
plt.show()
