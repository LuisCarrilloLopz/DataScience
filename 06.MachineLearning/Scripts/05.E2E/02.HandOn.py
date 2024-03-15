"""
Nombre del archivo: 02.HandOn.py
Autor: Luis Eduardo Carrillo López
Fecha de creación: 24/02/2024
Fecha de última modificación: 02/03/2024
"""

# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Leer el archivo de datos
df = pd.read_csv('https://raw.githubusercontent.com/eduardofc/data/main/titanic-2.csv')
df.drop({'PassengerId', 'Name', 'Ticket'}, axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.head()
#%%
df.shape
df.dtypes

#%%Verificar valores nulos
df.isnull().sum()

#%% Eliminar los dos valores de embarque que faltan
df.dropna(subset=['Embarked'], inplace=True)
df.isnull().sum()

#%%cambiar el valor de Cabin a si está informado ponga 'Yes' si no 'No'
df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df.drop('Cabin', axis=1, inplace=True)
df.head()

#%% Llenar los valore de Age con la distribución de la columna con inputers
from sklearn.impute import SimpleImputer
df_antiguo = df.copy()
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df['Age'].values.reshape(-1,1))

#%% Graficar la distribución de Age del df_antiguo y del df
plt.figure(figsize=(10,5))
sns.histplot(df_antiguo['Age'])
sns.histplot(df['Age'])
plt.show()

#%% probamos a hacer el inputer con un KNN inputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
df['Age'] = imputer.fit_transform(df['Age'].values.reshape(-1,1))

#%% Plots
cat_cols = df.select_dtypes(include='object').columns
num_cols= df.select_dtypes(include=('float64','int64')).columns

#%%
for col in num_cols:
    plt.figure(figsize=(10,5))
    sns.histplot(df[col])
    plt.title(col)
    plt.show()

#%%
for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, x=col)
    plt.title(col)
    plt.show()

#%% hacer una transformación logaritmica de la variable Fare para normalizar los datos
X1 = df[['Fare']]
X2 = df['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)

#Graficar la distribución de la variable 'Fare' de X1 y X2
plt.figure(figsize=(10,5))
sns.histplot(X1)
sns.histplot(X2)
plt.show()

#%% transformaciones
df['Fare'] = df['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)
df['SibSp'] = df['SibSp'].apply(lambda x: 2 if x > 0 else 0)
df['Parch'] = df['Parch'].apply(lambda x: np.log(x) if x > 0 else 0)
df['Title'] = df['Title'].apply(lambda x: 'Other' if x in ['Mr','Miss','Mrs'] else 'Otros')

#%% One hot encoder
from sklearn.preprocessing import OneHotEncoder

X = df[['Sex']]
encoder = OneHotEncoder(drop='if_binary')
X_new = encoder.fit_transform(X)

X = df[['Embarked']]
encoder = OneHotEncoder(drop='first')
X_new = encoder.fit_transform(X)

X = df[['Title']]
title_cats= [['Mr','Miss','Mrs']]
encoder = OneHotEncoder(categories=title_cats, handle_unknown='ignore')
X_new = encoder.fit_transform(X)

#%% Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

title_cats= [['Mr','Miss','Mrs']]
coltrans = ColumnTransformer([
    ('inputer', KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('encoder_sex', OneHotEncoder(drop='if_binary'), ['Sex']),
    ('encoder_embarked', OneHotEncoder(drop='first'), ['Embarked']),
    ('encoder_title', OneHotEncoder(categories=title_cats, handle_unknown='ignore'), ['Title'])
], remainder='passthrough')

coltrans.fit_transform(df)

#%% Ejercicio de modelado
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

coltrans = ColumnTransformer([
    ('inputer', KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('encoder_sex', OneHotEncoder(drop='if_binary'), ['Sex']),
    ('encoder_embarked', OneHotEncoder(drop='first'), ['Embarked']),
    ('encoder_title', OneHotEncoder(categories=title_cats, handle_unknown='ignore'), ['Title'])
], remainder='passthrough')

modelo = make_pipeline(coltrans, RandomForestClassifier())
modelo.fit(X_train, y_train)
modelo.score(X_test, y_test)

#%% Ejercicio2 de modelado
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

coltrans = ColumnTransformer([
    ('inputer', KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('encoder_sex', OneHotEncoder(drop='if_binary'), ['Sex']),
    ('encoder_embarked', OneHotEncoder(drop='first'), ['Embarked']),
    ('encoder_title', OneHotEncoder(categories=title_cats, handle_unknown='ignore'), ['Title'])
], remainder='passthrough')

modelo = Pipeline([('Coltransform Luis', coltrans), ('Algoritmo Luis', RandomForestClassifier())])
modelo.fit(X_train, y_train)
modelo.score(X_test, y_test)
modelo['Algoritmo Luis']

#%% Transformaciones numericas PorwerTransformer
from sklearn.preprocessing import PowerTransformer

X = df[['Fare']]
pt = PowerTransformer(method='yeo-johnson')
X_new = pt.fit_transform(X)

#%%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(X)
plt.subplot(1,2,2)
sns.histplot(X_new)
plt.show()

#%% Ejercicio 1 con PowerTransformer
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

coltrans = ColumnTransformer([
    ('inputer', KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
    ('encoder_sex', OneHotEncoder(drop='if_binary'), ['Sex']),
    ('encoder_embarked', OneHotEncoder(drop='first'), ['Embarked']),
    ('encoder_title', OneHotEncoder(categories=title_cats, handle_unknown='ignore'), ['Title']),
    ('power_transformer', PowerTransformer(), ['Fare','Age'])
], remainder='passthrough')

modelo = make_pipeline(coltrans, RandomForestClassifier())
modelo.fit(X_train, y_train)
modelo.score(X_test, y_test)

#%%
