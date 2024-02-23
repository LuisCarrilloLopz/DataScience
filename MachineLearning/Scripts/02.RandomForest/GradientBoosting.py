#%% Importar librerías
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

#%% Categorizar la variable objetivo
df['chd'] = df['chd'].apply(lambda x: 'Yes'  if x == 1 else 'No')
print(df['chd'].value_counts())
df[['famhist']] = pd.get_dummies(df[['famhist']],drop_first=True)

#%% Separar las variables predictoras y la variable de respuesta.
X = df.drop('chd', axis=1)
y = df['chd']

#%% Crear un conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Es importante que la distribución de la variable objetivo sea similar en train y test
print(f'La frecuencia de cada clase en train es: \n{y_train.value_counts(normalize=True)}')
print(f'La frecuencia de cada clase en test es:  \n{y_test.value_counts(normalize=True)}')

#%% Crear modelo de Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators = 450, subsample = 1,random_state = 123,n_iter_no_change = 10)
gb_classifier.fit(X_train, y_train)

#%% Evaluar el rendimiento del modelo
y_pred_gb_classifier = gb_classifier.predict(X_test)
accuracy_gb_classifier = accuracy_score(y_test, y_pred_gb_classifier)
print(f'Precisión del modelo con Gradient Boosting: {accuracy_gb_classifier}')

#%% se procede a observar el posible sobreajuste comparando predicciones en train y test.
y_pred_train = gb_classifier.predict(X_train)
y_pred_test = gb_classifier.predict(X_test)

print(f'Se tiene un accuracy para train de: {accuracy_score(y_train,y_pred_train)}')
print(f'Se tiene un accuracy para test de: {accuracy_score(y_test,y_pred_test)}')
print('Nótese la diferencia en accuracy para ambos conjuntos de datos y el posible sobreajuste.')

#%% Tuneo y evaluación del modelo para la variable dependiente categorica

param = {
    'n_estimators': [400,600],
    'n_iter_no_change': [None,5,10],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10, 50],
    'min_samples_leaf': [30]
}

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#%% recordar que arbol2 es el árbol cuyas VI son todas las variables.
# cv = crossvalidation
grid = GridSearchCV(gb_classifier, param, cv=4, scoring=scoring_metrics, refit='accuracy')
grid.fit(X_train, y_train)

#%%
results_grid = pd.DataFrame(grid.cv_results_)
# Mostrar resultados
print("Resultados de Grid Search:")
print(results_grid[["params", "mean_test_accuracy"]])

sorted_results = results_grid.sort_values(by='mean_test_accuracy', ascending=False).head(5)
print(sorted_results[["params", "mean_test_accuracy"]])

#%% se selecciona el modelo candidato, y se procede a analizar su robustez a lo largo de cross validation.
res_1 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[0]
res_2 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[1]
res_3 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[2]
res_4 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[3]
res_5 = sorted_results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[4]

#%%# Crear un boxplot para los cuatro valores de accuracy
plt.boxplot([res_1.values,res_2.values,res_3.values,res_4.values,res_5.values], labels = ['res_1','res_2','res_3','res_4','res_5'])
plt.title('Boxplots de Accuracy para los 4 Splits')
plt.xlabel('Splits de Cross Validation')
plt.ylabel('Accuracy')
plt.show()

#%% seleccionemos el segundo modelo dada su mayor robustez con respecto al propuesto por GridSearch
# nótese que "**" es para desempaquetar una lista de valores.

