"""
Nombre del archivo: 02.HandOn.py
Autor: Luis Eduardo Carrillo López
Fecha de creación: 24/02/2024
Fecha de última modificación: 02/03/2024
"""

# Importar librerías
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

#%% Cargar datos
df = pd.read_csv("https://raw.githubusercontent.com/eduardofc/data/main/diabetes.csv")
df.drop_duplicates(inplace=True)
df.head()

#%%
df.shape
df.groupby('Diabetes').size()

#%% Enfoque Clásico
seed = 15

models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('DTC', DecisionTreeClassifier(random_state=seed)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(random_state=seed)))

X = df.drop(columns="Diabetes")
y = df['Diabetes']

seed = 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

print("y_test:")
print("Número de 0s: ", np.count_nonzero(y_test==0))
print("Número de 1s: ", np.count_nonzero(y_test==1))

#%% Entrenar y evaluar modelos
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%%
confusion_matrix(y_true=y_test, y_pred=y_pred)

#%% Estrategia1: StratifiedKFolds
for name, model in models:
    cv_technique = StratifiedKFold(n_splits=5)
    grid_model = GridSearchCV(model, param_grid={}, cv=cv_technique, scoring="precision")

    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%% Estrategia 2: RandomOverSampling
ros = RandomOverSampler(random_state=seed)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("y_resampled:")
print("Número de 0s: ", np.count_nonzero(y_resampled==0))
print("Número de 1s: ", np.count_nonzero(y_resampled==1))

print("y_train:")
print("Número de 0s: ", np.count_nonzero(y_train==0))
print("Número de 1s: ", np.count_nonzero(y_train==1))

for name, model in models:
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%% Estrategia 3: RandomUnderSampling
rus = RandomUnderSampler(random_state=seed)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

print("y_resampled:")
print("Número de 0s: ", np.count_nonzero(y_resampled==0))
print("Número de 1s: ", np.count_nonzero(y_resampled==1))

print("y_train:")
print("Número de 0s: ", np.count_nonzero(y_train==0))
print("Número de 1s: ", np.count_nonzero(y_train==1))

for name, model in models:
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

# Estrategia 4: SMOTE
smote = SMOTE(random_state=seed)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("y_resampled:")
print("Número de 0s: ", np.count_nonzero(y_resampled==0))
print("Número de 1s: ", np.count_nonzero(y_resampled==1))

print("y_train:")
print("Número de 0s: ", np.count_nonzero(y_train==0))
print("Número de 1s: ", np.count_nonzero(y_train==1))

for name, model in models:
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    pre = precision_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%% Estrategia 5: Algoritmos desbalanceados
model = BalancedRandomForestClassifier(
    random_state=seed,
    replacement=True,
    max_depth=10,
    bootstrap=True,
    sampling_strategy='all'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
pre = precision_score(y_true=y_test, y_pred=y_pred)
rec = recall_score(y_true=y_test, y_pred=y_pred)
print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%% Estrategia 6: Class weight
model = RandomForestClassifier(
    n_estimators=3,
    max_depth=5,
    random_state=99,
    class_weight={0:0.5, 1:15}    # "balanced"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_true=y_test, y_pred=y_pred)
pre = precision_score(y_true=y_test, y_pred=y_pred)
rec = recall_score(y_true=y_test, y_pred=y_pred)
print(f"model: {name} \t acc: {acc:.2f} \t pre: {pre:.2f} \t rec: {rec:.2f}")

#%%