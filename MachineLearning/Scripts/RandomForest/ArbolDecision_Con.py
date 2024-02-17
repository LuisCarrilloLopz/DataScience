#%% Importar librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error,r2_score

#%% Macrovariables, rutas y parámetros

path = '/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/MachineLearning/Datasets/'
file = 'compress.csv'

# Obtener la extensión del archivo
_, ext = os.path.splitext(file)

# Leer el archivo dependiendo de su extensión
if ext == '.csv':
    df = pd.read_csv(os.path.join(path, file))
elif ext == '.xlsx':
    df = pd.read_excel(os.path.join(path, file))
else:
    print('Formato de archivo no soportado')

#%%
print(df.head())
print(df.columns)
print(df.shape)
print(df.isna().sum())
variables = df.columns

#%% Separar las variables predictoras y la variable de respuesta.
X_c = df.drop('cstrength', axis=1)
y_c = df['cstrength']

# criterion: Criterio de división: “squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default=”squared_error”.
# Se selecciona squared error con motivos ilustrativos. Es equivalente a la reducción de varianza.
# A priori, con motivos ilustrativos, se mantiene min_sample_split y max_depth en estos valores para facilitar la visualización del árbol.
# Recordar que estos son parámetros a modificar para encontrar el modelo óptimo.
# cpp_alpha es el parámetro de complejidad, el cual establece “penalizaciones” si se producen muchas divisiones. Cuanto más alto
# más pequeño será el árbol.
arbol3 = DecisionTreeRegressor(min_samples_split=30, criterion='squared_error', max_depth = 4, ccp_alpha = 0.01)

#%% Crear un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

# Es importante que la distribución de las clases sea 'similar' en train y test.
print(f'La media de la variable de respuesta en train es: \n{y_train.mean()}')
print(f'La media de la variable de respuesta en test es:  \n{y_test.mean()}')

#%% Construir el modelo
arbol3.fit(X_train, y_train)
print(pd.DataFrame({'nombre': arbol3.feature_names_in_, 'importancia': arbol3.feature_importances_}))

#%% Ordenar el DataFrame por importancia en orden descendente
df_importancia = pd.DataFrame({'Variable': arbol3.feature_names_in_, 'Importancia':
    arbol3.feature_importances_}).sort_values(by='Importancia', ascending=False)

# Crear un gráfico de barras
plt.bar(df_importancia['Variable'], df_importancia['Importancia'], color='skyblue')
plt.xlabel('Variable')
plt.ylabel('Importancia')
plt.title('Importancia de las características')
plt.xticks(rotation=45, ha='right') # Rotar los nombres en el eje x para mayor legibilidad
plt.tight_layout()

# Mostrar el gráfico
plt.show()

#%% Visualizar el árbol
plt.figure(figsize=(20, 20))
plot_tree(arbol3, filled=True, feature_names=X_c.columns)
plt.show()

#%% tuneo y evaluación del modelo para la variable dependiente continua
params_c = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_split': [5, 10, 20, 50, 100],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}

#%% Definir las metricas de evaluación
scoring_metrics_c = {'MAE': make_scorer(mean_absolute_error),
                     'MSE': make_scorer(mean_squared_error),
                     'RMSE': make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))}

# cv = crossvalidation
grid_search_c = GridSearchCV(estimator=arbol3, param_grid=params_c, cv=4, scoring=scoring_metrics_c, refit='MSE')
grid_search_c.fit(X_train, y_train)

#%% Obtener los resultados del grid search
results_c = pd.DataFrame(grid_search_c.cv_results_)

# Mostrar resultados
print("Resultados de Grid Search:")
print(results_c)

#%% Obtener el mejor modelo
best_model_c = grid_search_c.best_estimator_
print(grid_search_c.best_estimator_)

#%% Ajustar el mejor modelo con todo el conjunto de entrenamiento
best_model_c.fit(X_train, y_train)
# Predicciones en conjunto de entrenamiento y prueba
y_train_pred_c = best_model_c.predict(X_train)
y_test_pred_c = best_model_c.predict(X_test)
# Medidas de bondad de ajuste en train
y_pred_train_c = best_model_c.predict(X_train)
# Suponiendo que tienes los valores reales en y_test_c y
#las predicciones en y_pred_test_c
errores = y_train - y_pred_train_c
# Convertir los errores a un DataFrame
errores_df = pd.DataFrame({'Errores': errores})
# Box Plot de los errores en el conjunto de prueba
sns.boxplot(errores_df)
plt.title('Box Plot de Errores en el Conjunto de Prueba')
plt.show()

#%% Calcular diferentes medidas de bondad de ajuste
mae = mean_absolute_error(y_train, y_pred_train_c)
mse = mean_squared_error(y_train, y_pred_train_c)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred_train_c)
# Imprimir las métricas
print(f'MAE (Error Absoluto Medio): {mae:.2f}')
print(f'MSE (Error Cuadrático Medio): {mse:.2f}')
print(f'RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}')
print(f'R²: {r2:.2f}')

#%% Medidas de bondad de ajuste en test:
# Medidas de bondad de ajuste en train
y_pred_test_c = best_model_c.predict(X_test)
# Suponiendo que tienes los valores reales en y_test_c y
#las predicciones en y_pred_test_c
errores = y_test - y_pred_test_c
# Convertir los errores a un DataFrame
errores_df = pd.DataFrame({'Errores': errores})
# Box Plot de los errores en el conjunto de prueba
sns.boxplot(errores_df)
plt.title('Box Plot de Errores en el Conjunto de Prueba')
plt.show()

mae = mean_absolute_error(y_test, y_pred_test_c)
mse = mean_squared_error(y_test, y_pred_test_c)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test_c)

# Imprimir las métricas
print(f'MAE (Error Absoluto Medio): {mae:.2f}')
print(f'MSE (Error Cuadrático Medio): {mse:.2f}')
print(f'RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}')
print(f'R²: {r2:.2f}')

#%% Arbol
plt.figure(figsize=(20, 15))
plot_tree(best_model_c, feature_names=X_c.columns.tolist(), filled=True,
          proportion = True)
plt.show()

#%%
