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
file = 'arboles.csv'

# Obtener la extensión del archivo
_, ext = os.path.splitext(file)

# Leer el archivo dependiendo de su extensión
if ext == '.csv':
    df = pd.read_csv(os.path.join(path, file))
elif ext == '.xlsx':
    df = pd.read_excel(os.path.join(path, file))
else:
    print('Formato de archivo no soportado')

#%% Análisis exploratorio
print(df.head())
print(df.columns)
print(df.shape)
print(df.isna().sum())
variables = df.columns

#%% Categorizar la variable objetivo
df['chd'] = df['chd'].apply(lambda x: 'Yes'  if x == 1 else 'No')
print(df['chd'].value_counts())

#%% Separar las variables predictoras y la variable de respuesta.
X = df[['tobacco']]
y = df['chd']

#%% Creación de arbol de decisión
# min_sample_split: el número mínimo de casos que contiene una hoja para que pueda ser creada.
# criterion: Criterio de división: “gini”, “entropy”, “log_loss”.
# max_depth = Profundidad máxima del árbol. En caso de no especificar, el clasificador sigue segmentando hasta que las hojas son puras, o se alcanza el min_sample_split. Con caracter ilustrativo, se selecciona bajo.
arbol1 = DecisionTreeClassifier(min_samples_split=30, criterion='gini', max_depth = 2)

#%% Crear un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ES IMPORTANTE QUE LA DISTRIBUCIÓN DE LAS CLASES SEA 'SIMILAR' EN TRAIN Y TEST.
print(f'La frecuencia de cada clase en train es: \n{y_train.value_counts(normalize=True)}')
print(f'La frecuencia de cada clase en test es:  \n{y_test.value_counts(normalize=True)}')

#%% Construir el modelo
arbol1.fit(X_train, y_train)

#%% Conocer los niveles de la variable a predecir
print(arbol1.classes_)
# Conocer el nombre de las variables predictoras
print(arbol1.feature_names_in_)
# Obtener información detallada de cada nodo y las reglas de decisión
tree_rules = export_text(arbol1, feature_names=list(X.columns),show_weights=True)
print(tree_rules)

#%% Visualizar el árbol
plt.figure(figsize=(10, 10))
plot_tree(arbol1, filled=True, feature_names=X.columns, class_names=y.unique())
plt.show()
#%%
plt.figure(figsize=(10, 10))
plot_tree(arbol1, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], filled=True,proportion = True)
plt.show()


#%%# Se vuelve a entrenar el árbol con más variables. No necesariamente tiene que utilizar todas, por lo que es importante
# conocer la importancia predictiva de cada variable en el modelo.
#es importante tratar de forma adecuada las variables categóricas. Se convierten en numéricas con la regla: one hot encoding.
df[['famhist']] = pd.get_dummies(df[['famhist']],drop_first=True)

#%% Separar las variables predictoras y la variable de respuesta.
X = df.drop('chd', axis=1)
y = df['chd']

#%% Se selecciona profundidad 4 sólo con caracter ilustrativo, al simplificar el árbol.
arbol2 = DecisionTreeClassifier(min_samples_split=30, criterion='gini', max_depth = 4)

#%%#%% Crear un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ES IMPORTANTE QUE LA DISTRIBUCIÓN DE LAS CLASES SEA 'SIMILAR' EN TRAIN Y TEST.
print(f'La frecuencia de cada clase en train es: \n{y_train.value_counts(normalize=True)}')
print(f'La frecuencia de cada clase en test es:  \n{y_test.value_counts(normalize=True)}')

#%% Construir el modelo
arbol2.fit(X_train, y_train)

#%% Visualizar el árbol
plt.figure(figsize=(20, 20))
plot_tree(arbol2, filled=True, feature_names=X.columns, class_names=y.unique())
plt.show()

# %% Conocer los niveles de la variable a predecir
print(arbol2.classes_)
# Conocer el nombre de las variables predictoras
print(arbol2.feature_names_in_)
# Obtener información detallada de cada nodo y las reglas de decisión
tree_rules = export_text(arbol2, feature_names=list(X.columns),show_weights=True)
print(tree_rules)

#%% Se estudia la importancia - o valor predictivo - de cada variable en el modelo.
print(pd.DataFrame({'nombre': arbol2.feature_names_in_, 'importancia': arbol2.feature_importances_}))
# Ordenar el DataFrame por importancia en orden descendente
df_importancia = pd.DataFrame({'Variable': arbol2.feature_names_in_, 'Importancia':
    arbol2.feature_importances_}).sort_values(by='Importancia', ascending=False)
# Crear un gráfico de barras
plt.bar(df_importancia['Variable'], df_importancia['Importancia'], color='skyblue')
plt.xlabel('Variable')
plt.ylabel('Importancia')
plt.title('Importancia de las características')
plt.xticks(rotation=45, ha='right') # Rotar los nombres en el eje x para mayor legibilidad
plt.tight_layout()
# Mostrar el gráfico
plt.show()

#%% Tuneo y evaluación del modelo para la variable dependiente categorica

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_split': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#%% recordar que arbol2 es el árbol cuyas VI son todas las variables.
# cv = crossvalidation
grid_search = GridSearchCV(estimator=arbol2,param_grid=params,cv=4, scoring = scoring_metrics, refit='accuracy')
grid_search.fit(X_train, y_train)

#%% Obtener resultados del grid search
results = pd.DataFrame(grid_search.cv_results_)
# Mostrar resultados
print("Resultados de Grid Search:")
print(results[["params", "mean_test_accuracy",
                         "mean_test_precision_macro",
                         'mean_test_recall_macro', 'mean_test_f1_macro']])

#%% Obtener el mejor modelo
best_model = grid_search.best_estimator_
print(grid_search.best_estimator_)

#%% Para seleccionar una parametrización específica y la mejor de acuerdo con el criterio
# de GridSearch, acceder a esta y conocer su combinación.
results.iloc[8].params
# (En este caso, son ejemplos de selección aleatorios para ilustrar el script de selección y representación),.
# se selecciona el modelo candidato, y se procede a analizar su robustez a lo largo de cross validation.
res_1 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[2]
res_2 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[4]
res_3 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[18]

#%% Crear un boxplot para los cuatro valores de accuracy
plt.boxplot([res_1.values,res_2.values,res_3.values], labels = ["res_1","res_2","res_3"])
plt.title('Boxplots de Accuracy para los 4 Splits')
plt.xlabel('Splits de Cross Validation')
plt.ylabel('Accuracy')
plt.show()
# Nótese en la solución que boxplots con gran amplitud no son deseables,
# ya que se caracterizan por poca robustez de la solución

#%% # Obtener el mejor modelo (best estimador, o el seleccionado dado los pasos anteriores).
best_model = grid_search.best_estimator_
# Ajustar el mejor modelo con  todo el conjunto de entrenamiento
best_model.fit(X_train, y_train)
# Predicciones en conjunto de entrenamiento y prueba
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

#%% medidas de bondad de ajuste en train
conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Matriz de Confusión:")
print(conf_matrix)
print("\nMedidas de Desempeño:")
print(classification_report(y_train, y_train_pred))

#%% AUC en train
y_train_auc = pd.get_dummies(y_train,drop_first=True)

#%%  Calcular el área bajo la curva ROC (AUC)
y_prob_train = best_model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train_auc, y_prob_train)
roc_auc = auc(fpr, tpr)
print(f"\nÁrea bajo la curva ROC (AUC): {roc_auc:.2f}")
# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

#%% medidas de bondad de ajuste en test
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Matriz de Confusión:")
print(conf_matrix)
print("\nMedidas de Desempeño:")
print(classification_report(y_test, y_test_pred))

#%% AUC en test
y_test_auc = pd.get_dummies(y_test,drop_first=True)
# Calcular el área bajo la curva ROC (AUC)
y_prob_test = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_auc, y_prob_test)
roc_auc_test = auc(fpr, tpr)
print(f"\nÁrea bajo la curva ROC (AUC): {roc_auc:.2f}")
# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

#%% Arbol
plt.figure(figsize=(20, 15))
plot_tree(best_model, feature_names=X.columns.tolist(), filled=True,
          proportion = True)
plt.show()
#%%
