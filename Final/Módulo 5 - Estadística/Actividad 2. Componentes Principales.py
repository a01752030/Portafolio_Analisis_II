import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# 1. Leer el archivo y calcular matrices
data = pd.read_csv('paises_mundo.csv')
cov_matrix = data.cov()
corr_matrix = data.corr()
print("cov_matrix = ")
print(cov_matrix)
print("corr_matrix = ")
print(corr_matrix)
print(90*"*")

# 2. Valores y vectores propios
eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix)
eigenvalues_corr, eigenvectors_corr = np.linalg.eig(corr_matrix)

print("eigenvalues_corr = ")
print(eigenvalues_corr)
print("eigenvalues_cov = ")
print(eigenvalues_cov)
print("eigenvectors_cov = ")
print(eigenvectors_cov)
print("eigenvectors_corr = ")
print(eigenvectors_corr)


print(90*"*")


# 3. Proporción de varianza explicada
total_variance = np.sum(np.diag(cov_matrix))
explained_variance_ratio_cov = eigenvalues_cov / total_variance
explained_variance_ratio_corr = eigenvalues_corr / np.sum(eigenvalues_corr)

print("total_variance = ")
print(total_variance)
print("explained_variance_ratio_corr = ")
print(explained_variance_ratio_corr)
print("explained_variance_ratio_cov = ")
print(explained_variance_ratio_cov)
print(90*"*")


# 4. Acumule los resultados
cum_explained_variance_cov = np.cumsum(explained_variance_ratio_cov)
cum_explained_variance_corr = np.cumsum(explained_variance_ratio_corr)
print("cum_explained_variance_cov = ")
print(cum_explained_variance_cov)
print("cum_explained_variance_corr = ")
print(explained_variance_ratio_corr)

print(90*"*")

#PARTE II

# Graficar para la matriz S (varianza-covarianza)
plt.figure(figsize=(10, 6))
for i, variable in enumerate(data.columns):
    plt.arrow(0, 0, eigenvectors_cov[i, 0], eigenvectors_cov[i, 1], head_width=0.05, head_length=0.05, color='red')
    plt.text(eigenvectors_cov[i, 0] + 0.02, eigenvectors_cov[i, 1], variable)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title("PCA - Matriz S")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

# Graficar para la matriz R (correlaciones)
plt.figure(figsize=(10, 6))
for i, variable in enumerate(data.columns):
    plt.arrow(0, 0, eigenvectors_corr[i, 0], eigenvectors_corr[i, 1], head_width=0.05, head_length=0.05, color='blue')
    plt.text(eigenvectors_corr[i, 0] + 0.02, eigenvectors_corr[i, 1], variable)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title("PCA - Matriz R")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

#PARTE III

# Realizar PCA
pca = PCA()
pca_result = pca.fit_transform(data)

# Graficar las dos primeras componentes
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], color="blue", s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
plt.title("Visualización PCA")
plt.grid(True)
plt.show()

# Graficar la varianza explicada por cada componente principal
plt.figure(figsize=(10, 6))
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_*100, align='center', alpha=0.7)
plt.ylabel('Porcentaje de varianza explicada')
plt.xlabel('Componente principal')
plt.title('Gráfica de codo')
plt.show()

# Contribución de cada variable al PC1
contrib_pc1 = np.square(pca.components_[0]) * 100

plt.figure(figsize=(12, 6))
plt.bar(data.columns, contrib_pc1, align='center', alpha=0.7)
plt.ylabel('Contribución (%)')
plt.xlabel('Variables')
plt.title('Contribución de cada variable al PC1')
plt.xticks(rotation=90)
plt.show()