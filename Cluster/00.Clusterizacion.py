# Cargo las librerias que voy a utilizar
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
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

# %% Calcular la matriz de correlaciones y su representaci ́on gr ́afica: ¿Cu ́ales son las variables m ́as correlacionadas entre las caracter ́ısticas f ́ısicas de los pingu ̈inos?
plt.figure(figsize=(10, 8))
correlacion = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# %% Realizar un anáalisis de componentes principales (PCA)
# sobre la matriz de correlaciones, calculando un número adecuado de componentes (ḿaximo 4):
# Estudiar los valores de los autovalores obtenidos y las gŕaficas que los resumen. ¿Cúal es el número adecuado de componentes para representar eficientemente la variabilidad de las especies de pingüinos?

# Estandarizamos los datos
scaler = pd.DataFrame(
    StandardScaler().fit_transform(df.select_dtypes(include=[np.number])),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in variables],  # Nombres de columnas estandarizadas
    index=df.index  # Índices (etiquetas de filas) del DataFrame
)

# %% Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=4) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 4 para retener el maximo de las componentes principales (maximo= numero de variables).
pca = PCA(n_components=4)

# %% Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(notas_estandarizadas) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(scaler)

# %% Obtener los autovalores asociados a cada componente principal.:
autovalores = fit.explained_variance_

# %% Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T,
                            columns=['Autovector {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                            index=['{}_z'.format(variable) for variable in variables])

# %% Construimos las componentes
resultados_pca = pd.DataFrame(fit.transform(scaler),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                              index=scaler.index)

# %% Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_ * 100

# %% Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

# %% Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)])

# Imprimir la tabla
print(tabla)

# %% Representacion de la variabilidad explicada (Método del codo):
plot_varianza_explicada(var_explicada, fit.n_components_)

# %% Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca = PCA(n_components=2)
fit = pca.fit(scaler)

# %% Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# %% Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T,
                            columns=['Autovector {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                            index=['{}_z'.format(variable) for variable in variables])

# %% Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(scaler),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_ + 1)],
                              index=scaler.index)

# %% Añadimos las componentes principales a la base de datos estandarizada.
df_z_cp = pd.concat([scaler, resultados_pca], axis=1)

# %% Cálculo de las covarianzas y correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = df_z_cp.columns

# %% Guardamos el numero de componentes
n_variables = fit.n_features_in_

# %% Calcular la matriz de covarianzas entre veriables y componentes
Covarianzas_var_comp = df_z_cp.cov()
Covarianzas_var_comp = Covarianzas_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

# %% Calculo la matriz de correlaciones entre veriables y componentes
Correlaciones_var_comp = df_z_cp.corr()
Correlaciones_var_comp = Correlaciones_var_comp.iloc[:fit.n_features_in_, fit.n_features_in_:]

# %% Representación gráfica de la matriz de covarianzas entre variables y componentes
cos2 = Correlaciones_var_comp ** 2
plot_cos2_heatmap(cos2)

# %% Cantidad total de variabildiad explicada de una variable
# por el conjunto de componentes

plot_cos2_bars(cos2)

# %% Contribuciones de cada variable en la construcción de las componentes
contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2, autovalores, fit.n_components)

# %% Representación de las correlaciones entre variables y componentes
plot_corr_cos(fit.n_components, Correlaciones_var_comp)

# %% Nube de puntos de las observaciones en las componentes = ejes
plot_pca_scatter(pca, scaler, fit.n_components)

# %% Segunda parte de la tarea

# Creamos el mapa de calor (heatmap) donde se representan de manera ordenada los
# valores observados, así como un proceso de cluster jerárquico donde se muestran
# los diferentes pasos iterativos de unión de observaciones.
sns.clustermap(df.select_dtypes(include=[np.number]), cmap='coolwarm', annot=True)
# Agregamos un título al gráfico
plt.title('Mapa de Calor')
# Etiquetamos el eje x
plt.xlabel('Pingüinos')
# Etiquetamos el eje y
plt.ylabel('Tipo de Pingüino')
# Mostramos el gráfico
plt.show()

data = df.select_dtypes(include=[np.number]).copy()

# %% Calculamos distancias sin estandarizar
# Calcula la matriz de distancias Euclidianas entre las observaciones
distance_matrix = distance.cdist(data, data, 'euclidean')
# Crea un DataFrame a partir de la matriz de distancias con los índices de df
distance_df = pd.DataFrame(distance_matrix, index=data.index, columns=data.index)
# La distance_matrix es una matriz 2D que contiene las distancias Euclidianas
# entre todas las parejas de observaciones.

# %% Seleccionamos las primeras 5 filas y columnas de la matriz de distancias
distance_small = distance_matrix[:5, :5]
# Agregamos los nombres de índice a la matriz de distancias
distance_small = pd.DataFrame(distance_small, index=data.index[:5], columns=df.index[:5])
# Redondeamos los valores en la matriz de distancias
distance_small_rounded = distance_small.round(2)
print("Matriz de Distancias Redondeada:", distance_small_rounded)

# %% Representamos gráficamente la matriz de distancias
# Crea una nueva figura para el gráfico con un tamaño específico
plt.figure(figsize=(10, 8))
# Genera un mapa de calor usando Seaborn
# - `distance_df`: DataFrame de pandas que contiene los datos para el mapa de calor.
# - `annot=False`: No muestra las anotaciones (valores de los datos) en las celdas del mapa.
# - `cmap="YlGnBu"`: Utiliza la paleta de colores "Yellow-Green-Blue" para el mapa de calor.
# - `fmt=".1f"`: Formato de los números en las anotaciones, en este caso no se usan.
sns.heatmap(distance_df, annot=False, cmap="YlGnBu", fmt=".1f")
# Muestra el gráfico
plt.show()

# %% Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix).
# Clustermap es una función compleja que combina un mapa de calor con dendrogramas para mostrar la agrupación de datos.
# Aquí, estamos usando el dataframe 'distance_df' que contiene las distancias euclidianas entre las observaciones.
# La opción 'cmap' establece la paleta de colores a 'YlGnBu', que es un gradiente de azules y verdes.
# La opción 'fmt' se usa para formatear las anotaciones numéricas, en este caso a un decimal.
# La opción 'annot=False' indica que no queremos anotaciones numéricas en las celdas del mapa de calor.
# La opción 'method' especifica el método de agrupamiento a usar, en este caso 'average' que calcula la media de las distancias.
linkage = sns.clustermap(distance_df, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row

# %% Estandarizamos los datos
# Inicializamos el escalador de estandarizacion
scaler = StandardScaler()

# Ajustamos y transformamos el DataFrame para estandarizar las columnas
# 'fit_transform' primero calcula la media y la desviacion estandar de cada columna para luego realizar la estandarizacion.
df_std = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Asignamos el indice del DataFrame original 'df' al nuevo DataFrame 'df_std'
# Esto es importante para mantener la correspondencia de los indices de las filas despues de la estandarizacion.
df_std.index = data.index

# %% Calculamos las distancias euclidianas por pares entre las filas del DataFrame estandarizado
# 'cdist' calcula la distancia euclidiana entre cada par de filas en 'df_std'.
# Esto resulta en una matriz de distancias donde cada elemento [i, j] es la distancia entre la fila i y la fila j.
distance_std = distance.cdist(df_std, df_std, 'euclidean')

# Imprimimos los primeros 5x5 elementos de la matriz de distancias para tener una vista previa
print(distance_std[:5, :5].round(2))

# %% # Esto determina las dimensiones del grafico
plt.figure(figsize=(10, 8))

# Creamos un nuevo DataFrame para la matriz de distancias
# 'distance_std' se convierte en un DataFrame con indices y columnas correspondientes a 'df_std'
# Esto facilita la interpretacion del mapa de calor, ya que las filas y columnas estaran etiquetadas con los indices de 'df_std'
df_std_distance = pd.DataFrame(distance_std, index=df_std.index, columns=df_std.index)

# Generamos un mapa de calor utilizando Seaborn
# - 'df_std_distance': DataFrame que contiene los datos de distancia para visualizar.
# - 'annot=False': No muestra anotaciones numericas en cada celda del mapa de calor.
# - 'cmap="YlGnBu"': Define una paleta de colores en tonos de azul y verde para el mapa de calor.
# - 'fmt=".1f"': Formato de los numeros en las anotaciones, en este caso, un decimal.
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")

# Mostramos el grafico resultante
plt.show()

# %% Realizamos clustering jerárquico para obtener la matriz de enlace (linkage matrix).
linkage = sns.clustermap(df_std_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='ward').dendrogram_row

# Calculamos la matriz de enlace (linkage matrix)
# 'sch.linkage' realiza el clustering jerarquico
# 'method='ward'' es uno de los metodos de enlace, que minimiza la varianza de los clusters que se van fusionando
linkage_matrix = sch.linkage(df_std_distance,
                             method='ward')  # Puedes elegir un metodo de enlace diferente si es necesario

# Creamos el dendrograma
# 'sch.dendrogram' dibuja el dendrograma basado en la 'linkage_matrix'
# 'labels=df_std_distance.index.tolist()' usa los indices del DataFrame como etiquetas para las hojas del dendrograma
# 'leaf_rotation=90' rota las etiquetas de las hojas para mejorar la legibilidad
dendrogram = sch.dendrogram(linkage_matrix, labels=df_std_distance.index.tolist(), leaf_rotation=90)

# Mostramos el dendrograma
plt.show()

# %% # Establecemos un umbral de color para el dendrograma
color_threshold = 200

# Creamos el dendrograma con el umbral de color especificado
dendrogram = sch.dendrogram(linkage_matrix, labels=df_std_distance.index.tolist(), leaf_rotation=90,
                            color_threshold=color_threshold)

# Mostramos el dendrograma
plt.show()

# %% Asignamos las observaciones de datos a 4 clusters

# Especificamos el numero de clusters a formar
num_clusters = 3

# Asignamos los datos a los clusters
# 'fcluster' forma clusters planos a partir de la matriz de enlace 'linkage_matrix'
# 'criterion='maxclust'' significa que formaremos un numero maximo de 'num_clusters' clusters
cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Mostramos las asignaciones de clusters
print("Cluster Assignments:", cluster_assignments)

# Creamos una nueva columna 'Cluster4' y asignamos los valores de 'cluster_assignments' a ella
# Ahora 'df' contiene una nueva columna 'Cluster4' con las asignaciones de cluster
data['Cluster4'] = cluster_assignments

# Visualización de la distribución espacial de los clusters
# Paso 1: Realizar PCA
pca = PCA(n_components=2)  # Inicializamos PCA para 2 componentes principales
principal_components = pca.fit_transform(data)  # Transformamos los datos a 2 componentes

# Creamos un nuevo DataFrame para los componentes principales 2D
# Nos aseguramos de que df_pca tenga el mismo índice que df
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data.index)

# Paso 2: Crear un gráfico de dispersión con colores para los clusters
plt.figure(figsize=(10, 6))  # Establecemos el tamaño del gráfico

# %% Recorremos las asignaciones únicas de clusters y trazamos puntos de datos con el mismo color
for cluster in np.unique(cluster_assignments):
    cluster_indices = df_pca.loc[cluster_assignments == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Etiqueta para cada cluster
    # Anotamos cada punto con el nombre del país
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']), fontsize=10,
                     textcoords="offset points",  # cómo posicionar el texto
                     xytext=(0, 10),  # distancia del texto a los puntos (x,y)
                     ha='center')  # alineación horizontal puede ser izquierda, derecha o centro

plt.title("Gráfico de PCA 2D con Asignaciones de Cluster")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()

# %%# Metodo del codo
# Creamos un array para almacenar los valores de WCSS para diferentes valores de K
wcss = []

for k in range(1, 11):  # Puedes elegir un rango diferente de valores de K
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  # Inserta es el valor de WCSS

# Graficamos los valores de WCSS frente al numero de grupos (K) y buscamos el punto "codo"
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Metodo del Codo')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# %% Metodo de la silueta
# Creamos un array para almacenar los puntajes de silueta para diferentes valores de K
silhouette_scores = []

# Ejecutamos el clustering K-means para un rango de valores de K y calculamos el puntaje de silueta para cada K
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)

# Graficamos los puntajes de silueta frente al numero de clusters (K)
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Metodo de la Silueta')
plt.xlabel('Numero de Clusters (K)')
plt.ylabel('Puntaje de Silueta')
plt.grid(True)
plt.show()

# %% Analisis no jerarquico
# Configurar el número de clusters (k=4)
k = 2

# Inicializar el modelo KMeans
# 'n_clusters=k' indica el número de clusters a formar
# 'random_state=0' asegura la reproducibilidad de los resultados
# 'n_init=10' indica el número de veces que el algoritmo se ejecutará con diferentes centroides iniciales
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)

# Ajustar el modelo KMeans a los datos estandarizados
# 'df_std' es el DataFrame que contiene los datos previamente estandarizados
kmeans.fit(df_std)

# Obtener las etiquetas de los clusters para los datos
# 'kmeans.labels_' contiene la asignación de cada punto a un cluster
kmeans_cluster_labels = kmeans.labels_

# Creamos una nueva columna 'Cluster' y asignamos los valores de 'kmeans_cluster_labels' a ella
# 'Cluster4_v2' sera el nombre de la nueva columna en el DataFrame 'df'
data['Cluster2_v2'] = kmeans_cluster_labels

# Ahora 'df' contiene una nueva columna 'Cluster4_v2' con las asignaciones de cluster
# Imprimimos los valores de la columna 'Cluster4_v2' para verificar las asignaciones de cluster
print(data["Cluster2_v2"])

# %% Visualizacion de la distribucion espacial de los clusters
# Paso 1: Realizar PCA
pca = PCA(n_components=2)  # Inicializar PCA para reducir a 2 componentes principales
principal_components = pca.fit_transform(data)  # Ajustar y transformar los datos

# Crear un nuevo DataFrame para los componentes principales en 2D
# Se asignan nombres a las columnas como 'PC1' y 'PC2' y se mantiene el mismo indice que 'df'
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)

# Paso 2: Crear un grafico de dispersion con colores para los clusters de KMeans
plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura

# Iterar a traves de las etiquetas unicas de clusters y graficar puntos de datos con el mismo color
for cluster in np.unique(kmeans_cluster_labels):
    cluster_indices = df_pca.loc[kmeans_cluster_labels == cluster].index
    plt.scatter(df_pca.loc[cluster_indices, 'PC1'],
                df_pca.loc[cluster_indices, 'PC2'],
                label=f'Cluster {cluster}')  # Poner una etiqueta para cada cluster

    # Anotar cada punto con su respectivo indice
    for i in cluster_indices:
        plt.annotate(i,
                     (df_pca.loc[i, 'PC1'], df_pca.loc[i, 'PC2']), fontsize=10,
                     textcoords="offset points",  # Define como se posicionara el texto
                     xytext=(0, 10),  # Define la distancia del texto a los puntos (x,y)
                     ha='center')  # Define la alineacion horizontal del texto

# Configurar el titulo y las etiquetas de los ejes del grafico
plt.title("Grafico 2D de PCA con Asignaciones de Cluster KMeans")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()  # Mostrar la leyenda
plt.grid()  # Mostrar la cuadricula
plt.show()  # Mostrar el grafico

# %% Configuramos el modelo KMeans con 4 clusters y un estado aleatorio fijo
kmeans = KMeans(n_clusters=2, random_state=0)
# Ajustamos el modelo KMeans a los datos estandarizados
kmeans.fit(df_std)
# Obtenemos las etiquetas de clusters resultantes
labels = kmeans.labels_

# Calculamos los valores de silueta para cada observación
silhouette_values = silhouette_samples(df_std, labels)

# Configuramos el tamaño de la figura para el gráfico
plt.figure(figsize=(8, 6))
y_lower = 10  # Inicio del margen inferior en el gráfico

# Iteramos sobre los 4 clusters para calcular los valores de silueta y dibujar el gráfico
for i in range(4):
    # Extraemos los valores de silueta para las observaciones en el cluster i
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    # Ordenamos los valores para que el gráfico sea más claro
    ith_cluster_silhouette_values.sort()

    # Calculamos donde terminarán las barras de silueta en el eje y
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    # Elegimos un color para el cluster
    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    # Rellenamos el gráfico entre un rango en el eje y con los valores de silueta
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    # Etiquetamos las barras de silueta con el número de cluster en el eje y
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Actualizamos el margen inferior para el siguiente cluster
    y_lower = y_upper + 10  # 10 para el espacio entre clusters

# Títulos y etiquetas para el gráfico
plt.title("Gráfico de Silueta para los Clusters")
plt.xlabel("Valores del Coeficiente de Silueta")
plt.ylabel("Etiqueta del Cluster")
plt.grid(True)  # Añadimos una cuadrícula para mejor legibilidad
plt.show()  # Mostramos el gráfico resultante

# %% Caracterizamos cada cluster
# Añadimos las etiquetas como una nueva columna al DataFrame original
data['Cluster'] = labels
# Ordenamos el DataFrame por la columna "Cluster"
df_sort = data.sort_values(by="Cluster")

# Agrupamos los datos por la columna 'Cluster' y calculamos la media de cada grupo
# Esto proporcionará las coordenadas de los centroides de los clusters en el espacio de los datos originales
cluster_centroids_orig = df_sort.groupby('Cluster').mean()
# Redondeamos los valores para facilitar la visualización
cluster_centroids_orig.round(2)
# %% 'cluster_centroids_orig' ahora contiene los centroides de cada cluster en el espacio de los datos originales

