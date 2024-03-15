"""
Nombre del archivo: 00.TimeSeries.py
Autor: Luis Eduardo Carrillo López
Fecha de creación: 24/02/2024
Fecha de última modificación: 24/02/2024
"""

import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from pickle import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#%% Load the digits dataset
datos= load_digits(as_frame=True)
x=datos.data
y=datos.target

#%% Modelo 1 Logistig regression
modelo1= LogisticRegression()
modelo1.fit(x,y)

#%%
modelo1.predict(x)
modelo1.score(x,y)

#%% Save the model
dump(modelo1,open("/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/0.6MachineLearning/Datasets/00.Logistic_Regresion.pkl", "wb"))

#%% Load the model
modelo2 = load(open("/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/0.6MachineLearning/Datasets/00.Logistic_Regresion.pkl", "rb"))

#%% Modelo 2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

#%%
model2 = LogisticRegression()
model2.fit(X_train, y_train)
model2.score(X_test, y_test)

#%%Modelo 3
scaler = StandardScaler()
algoritmo = LogisticRegression()
modelo3 = make_pipeline(scaler, algoritmo)
modelo3.fit(X_train, y_train)
modelo3.score(X_test, y_test)
#%%
modelo3.steps[1][1].coef_
modelo3.steps[1][1].intercept_

#%% Comparación de algoritmos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%% Crear los modelos
modelos = []
modelos.append(('LR', LogisticRegression()))
modelos.append(('RF', RandomForestClassifier()))
modelos.append(('SVM', SVC()))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('CART', DecisionTreeClassifier()))
modelos.append(('NB', GaussianNB()))
modelos.append(('LDA', LinearDiscriminantAnalysis()))

#%% Evaluar cada modelo
for nombre, modelo in modelos:
    modelo.fit(X_train, y_train)
    print(f'{nombre}: {modelo.score(X_test, y_test)}')
    modelo.score(X_test, y_test)

#%% Comparación de algoritmos con semilla
modelos = []
modelos.append(('LR', LogisticRegression(random_state=1234)))
modelos.append(('RF', RandomForestClassifier(random_state=1234)))
modelos.append(('SVM', SVC(random_state=1234)))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('CART', DecisionTreeClassifier(random_state=1234)))
modelos.append(('NB', GaussianNB()))
modelos.append(('LDA', LinearDiscriminantAnalysis()))

#%% Evaluar cada modelo
for nombre, modelo in modelos:
    modelo.fit(X_train, y_train)
    print(f'{nombre}: {modelo.score(X_test, y_test)}')
    modelo.score(X_test, y_test)



