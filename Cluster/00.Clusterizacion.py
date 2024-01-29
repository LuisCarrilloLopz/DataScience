# Cargo las librerias que voy a utilizar
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Data.FuncionesMineria import plot_varianza_explicada, plot_cos2_heatmap, plot_cos2_bars, plot_corr_cos, \
    plot_pca_scatter, plot_contribuciones_proporcionales

# %% Establecemos nuestro escritorio de trabajo
os.chdir('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Cluster/Data')

# %% Cargo los datos

df = pd.read_excel('penguins.xlsx')

with open('FuncionesMineria.py') as file:
    exec(file.read())

variables = list(df.select_dtypes(include=[np.number]).columns)

#%% Calcular la matriz de correlaciones y su representaci ́on gr ́afica: ¿Cu ́ales son las variables m ́as correlacionadas entre las caracter ́ısticas f ́ısicas de los pingu ̈inos?
plt.figure(figsize=(10, 8))
correlacion = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

#%% Realizar un anáalisis de componentes principales (PCA)
# sobre la matriz de correlaciones, calculando un número adecuado de componentes (ḿaximo 4):
# Estudiar los valores de los autovalores obtenidos y las gŕaficas que los resumen. ¿Cúal es el número adecuado de componentes para representar eficientemente la variabilidad de las especies de pingüinos?

# Estandarizamos los datos
scaler = pd.DataFrame(
    StandardScaler().fit_transform(df.select_dtypes(include=[np.number])),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in variables],  # Nombres de columnas estandarizadas
    index=df.index  # Índices (etiquetas de filas) del DataFrame
)

#%% Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=4) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 4 para retener el maximo de las componentes principales (maximo= numero de variables).
pca = PCA(n_components=4)

#%% Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(notas_estandarizadas) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(scaler)

#%% Obtener los autovalores asociados a cada componente principal.:
autovalores = fit.explained_variance_

#%% Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T,
                            columns=['Autovector {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                            index=['{}_z'.format(variable) for variable in variables])

#%% Construimos las componentes
resultados_pca = pd.DataFrame(fit.transform(scaler),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                              index=scaler.index)

#%% Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_ * 100

#%% Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

#%% Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)])

# Imprimir la tabla
print(tabla)

#%% Representacion de la variabilidad explicada (Método del codo):
plot_varianza_explicada(var_explicada, fit.n_components_)

#%% Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca = PCA(n_components=2)
fit = pca.fit(scaler)

#%% Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

#%% Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T,
                            columns=['Autovector {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                            index=['{}_z'.format(variable) for variable in variables])

#%% Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(scaler),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                              index=scaler.index)

#%% Añadimos las componentes principales a la base de datos estandarizada.
df_z_cp = pd.concat([scaler, resultados_pca], axis=1)

#%% Cálculo de las covarianzas y correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = df_z_cp.columns

#%% Guardamos el numero de componentes
n_variables = fit.n_features_in_

#%% Calcular la matriz de covarianzas entre veriables y componentes
Covarianzas_var_comp = df_z_cp.cov()
Covarianzas_var_comp = Covarianzas_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

#%% Calculo la matriz de correlaciones entre veriables y componentes
Correlaciones_var_comp = df_z_cp.corr()
Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

# %% Representación gráfica de la matriz de covarianzas entre variables y componentes
cos2 = Correlaciones_var_comp ** 2
plot_cos2_heatmap(cos2)

#%% Cantidad total de variabildiad explicada de una variable
# por el conjunto de componentes

plot_cos2_bars(cos2)

# %% Contribuciones de cada variable en la construcción de las componentes
contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2, autovalores, fit.n_components)

#%% Representación de las correlaciones entre variables y componentes
plot_corr_cos(fit.n_components, Correlaciones_var_comp)

#%% Nube de puntos de las observaciones en las componentes = ejes
plot_pca_scatter(pca, scaler, fit.n_components)

#%%
