#%% Importar librerías
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

#%% Crear modelo de Random Forest
random_forest = RandomForestClassifier(n_estimators = 60,bootstrap = True, max_depth = 20, min_samples_split=10, criterion='entropy',min_samples_leaf = 10,random_state=123)
random_forest.fit(X_train, y_train)

#%% Evaluar el rendimiento del modelo
y_pred_random_forest = random_forest.predict(X_test)
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
print(f'Precisión del modelo con Random Forest: {accuracy_random_forest}')

#%% se procede a observar el posible sobreajuste comparando predicciones en train y test.
y_pred_train = random_forest.predict(X_train)
y_pred_test = random_forest.predict(X_test)

print(f'Se tiene un accuracy para train de: {accuracy_score(y_train,y_pred_train)}')
print(f'Se tiene un accuracy para test de: {accuracy_score(y_test,y_pred_test)}')
print('Nótese la diferencia en accuracy para ambos conjuntos de datos y el posible sobreajuste.')

#%% Tuneo y evaluación del modelo para la variable dependiente categorica
param = {
    'n_estimators': [3,10, 50, 100,250],
    'max_depth': [2,5, 10, 20, 30],
    'bootstrap': [True, False],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

#%% recordar que arbol2 es el árbol cuyas VI son todas las variables.
# cv = crossvalidation
grid_search_RF = GridSearchCV(random_forest, param, cv=4, scoring=scoring_metrics, refit='accuracy')
grid_search_RF.fit(X_train, y_train)

#%%Obtener el mejor modelo
best_model_RF = grid_search_RF.best_estimator_
print(grid_search_RF.best_estimator_)

#%% se procede a observar el posible sobreajuste comparando predicciones en train y test.
y_pred_train_RF = best_model_RF.predict(X_train)
y_pred_test_RF = best_model_RF.predict(X_test)

print(f'Se tiene un accuracy para train de: {accuracy_score(y_train,y_pred_train_RF)}')
print(f'Se tiene un accuracy para test de: {accuracy_score(y_test,y_pred_test_RF)}')
print('Comprobar que la diferencia no sea muy grande por temas de sobreajuste')

#%%
results_RF = pd.DataFrame(grid_search_RF.cv_results_)
# Mostrar resultados
print("Resultados de Grid Search:")
print(results_RF[["params", "mean_test_accuracy"]])

sorted_results = results_RF.sort_values(by='mean_test_accuracy', ascending=False).head(5)
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
random_f_2 = RandomForestClassifier(**sorted_results['params'].iloc[2],random_state=123)
random_f_2.fit(X_train, y_train)
res_rf_2 = random_f_2.predict(X_test)
accuracy_score(y_test,res_rf_2) # 0.698

#%% Si se quiere conocer quién tiene mayor robustez en sensibilidad por cuestiones de criterio exógeno:
sorted_results[['std_test_recall_macro']]

#%% se procede a observar el posible sobreajuste comparando predicciones en train y test.
y_pred_train_RF_2 = random_f_2.predict(X_train)
y_pred_test_RF_2 = random_f_2.predict(X_test)

print(f'Se tiene un accuracy para train de: {accuracy_score(y_train,y_pred_train_RF_2)}')
print(f'Se tiene un accuracy para test de: {accuracy_score(y_test,y_pred_test_RF_2)}')

#%%
