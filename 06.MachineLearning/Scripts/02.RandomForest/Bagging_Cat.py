#%% Importar librerías
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier, export_text,
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

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
df[['famhist']] = pd.get_dummies(df[['famhist']],drop_first=True)
#%% Separar las variables predictoras y la variable de respuesta.
X = df.drop('chd', axis=1)
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

#%% Crear el modelo de bagging
# n_estimators: número de árboles que se van a crear.
# max_samples: número de muestras que se van a tomar para crear cada árbol.
# max_features: número de variables que se van a tomar para crear cada árbol.
# bootstrap: si se van a tomar muestras con reemplazo o no.
# n_jobs: número de núcleos que se van a utilizar para crear los árboles.
# random_state: semilla para reproducibilidad.
bagging = BaggingClassifier(arbol1,max_samples = 300, max_features = 9,n_estimators=250, random_state=123, oob_score = True)
bagging.fit(X_train, y_train)

#%%Tuneo y evaluación del modelo para la variable dependiente categorica

param = {
    'n_estimators': [10, 50, 100,250], 'max_samples': [1,75,150,300], 'max_features': [1,4,7,9], 'bootstrap': [True, False], 'bootstrap_features': [True, False]
}

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#%% recordar que arbol2 es el árbol cuyas VI son todas las variables.
# cv = crossvalidation
grid_search = GridSearchCV(bagging, param, cv=5, scoring=scoring_metrics, refit='accuracy')
grid_search.fit(X_train, y_train)

#%% Obtener resultados del grid search
results = pd.DataFrame(grid_search.cv_results_)
# Mostrar resultados
print("Resultados de Grid Search:")
print(results[["params", "mean_test_accuracy"]])

results_sorted = results[["params", "mean_test_accuracy"]].sort_values(by="mean_test_accuracy", ascending=False)
print(results_sorted)

#%% Obtener el mejor modelo
best_model = grid_search.best_estimator_
print(grid_search.best_estimator_)

# Obtener los tres mejores modelos
top_3_models = results_sorted.head(3)

# Crear instancias de BaggingClassifier para cada modelo
for i, row in top_3_models.iterrows():
    model = BaggingClassifier(**row['params'])
    model.fit(X_train, y_train)
    print(model)

#%% Comparar accuracy de los tres modelos arbol1, bagging y best_model
res_1 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[84]
res_2 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[100]
res_3 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[102]

#%%#%% Crear un boxplot para los cuatro valores de accuracy
plt.boxplot([res_1.values,res_2.values,res_3.values], labels = ["res_1","res_2","res_3"])
plt.title('Boxplots de Accuracy para los 4 Splits')
plt.xlabel('Splits de Cross Validation')
plt.ylabel('Accuracy')
plt.show()
# Nótese en la solución que boxplots con gran amplitud no son deseables,
# ya que se caracterizan por poca robustez de la solución

#%% Comparar el accuracy de los tres modelos arbol1, bagging y best_model
print(f'Accuracy arbol1: {arbol1.score(X_test, y_test)}')
print(f'Accuracy bagging: {bagging.score(X_test, y_test)}')
print(f'Accuracy best_model: {best_model.score(X_test, y_test)}')

#%% Comparar el accuracy de los tres modelos arbol1, bagging y best_model OTRA FORMA
y_pred_arbol1 = arbol1.predict(X_test)
y_pred_bagging = bagging.predict(X_test)
y_pred_best_model = best_model.predict(X_test)

#Evaluar rendimiento del modelo
accuracy_arbol1 = accuracy_score(y_test, y_pred_arbol1)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
accuracy_best_model = accuracy_score(y_test, y_pred_best_model)

print(f'Precisión del modelo con arbol1: {accuracy_arbol1}')
print(f'Precisión del modelo con bagging: {accuracy_bagging}')
print(f'Precisión del modelo con best_model: {accuracy_best_model}')

#%% graficar el accuracy de los tres modelos
accuracy = [arbol1.score(X_test, y_test), bagging.score(X_test, y_test), best_model.score(X_test, y_test)]
models = ['arbol1', 'bagging', 'best_model']
plt.bar(models, accuracy)
plt.xlabel('Modelos')
plt.ylabel('Accuracy')
plt.title('Comparación de Accuracy entre los modelos')
plt.show()

