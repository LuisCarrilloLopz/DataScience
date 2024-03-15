"""
Nombre del archivo: 00.TimeSeries.py
Autor: Luis Eduardo Carrillo López
Fecha de creación: 24/02/2024
Fecha de última modificación: 24/02/2024
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

sns.set(style="darkgrid")
import warnings
warnings.filterwarnings('ignore')

#%% Importing the dataset
df = pd.read_csv('https://raw.githubusercontent.com/eduardofc/data/main/breast_cancer_data.csv')
df.drop(columns=["Unnamed: 32","id"],inplace=True)
df.drop_duplicates(inplace=True)
df.head()

#%% Exploratory Data Analysis
df.shape
df.dtypes
df.describe()
df.corr(numeric_only=True)
df.groupby('diagnosis').size()

#%% Plots (EDA)
sns.histplot(df ,x = 'radius_mean', kde=True, bins=30, hue='diagnosis')
plt.show()

#%%
sns.boxplot(df, x='texture_worst', y='diagnosis', hue='diagnosis')
plt.show()
#%%
sns.violinplot(df, x='texture_worst', y='diagnosis', hue='diagnosis')
plt.show()
#%%sacar las variables numericas del df en una lista
numericas = df.select_dtypes(include=[np.number]).columns.tolist()

#%%
for c in numericas:
    plt.subplot(1,2,1)
    sns.histplot(df, x=c, kde=True, bins=30, hue='diagnosis')
    plt.subplot(1,2,2)
    sns.boxplot(df, x=c, y=c, hue='diagnosis')
    plt.show()

#%%
plt.figure(figsize=(3,3))
sns.jointplot(df,x="perimeter_worst",y="texture_worst",hue="diagnosis")
plt.show()

#%%
plt.figure(figsize=(15,13))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

#%%
plt.figure(figsize=(10,3))
sns.countplot(df, x='diagnosis')
plt.show()
#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
X = df[['area_mean']]
print(X.shape)
#X=df[['area_mean','radius_mean']]
#print(X.shape)

#%%
X = df[['area_mean']]
scaler= StandardScaler()
X_new= scaler.fit_transform(X)
#%%
X = df[['area_mean']]
scaler= MinMaxScaler()
X_new= scaler.fit_transform(X)
a=scaler.inverse_transform
#%%
X = df[['area_mean']]
scaler=Binarizer(threshold=np.median(X))
X_new=scaler.fit_transform(X)

#%%
from sklearn.feature_selection import SelectKBest, chi2
X = df[numericas]
y= df['diagnosis']
selector = SelectKBest(chi2, k=10)
selector.fit(X,y)
X_new= selector.transform(X)
#%%
pd.DataFrame(zip(X.columns,selector.pvalues_),columns=['feat','pval']).sort_values(by='pval').head(5)

#%%
print(X.shape)
print(X_new.shape)

#%% PCA
from sklearn.decomposition import PCA
X = df[numericas]
pca = PCA(n_components=10)
pca.fit(X)
X_new = pca.transform(X)

#%%Ejercicio
# Logistic Regression + PCA(10)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

seed = 99
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=seed)

#%%
for i in range(1,10):
    selector = SelectKBest(score_func=chi2, k=i)
    scaler = MinMaxScaler()
    alg = LogisticRegression(random_state=seed)
    model = make_pipeline(scaler, selector, alg)

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"i={i} \t acc={acc:.4f}")

#%%""" RFE """
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=seed)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

#%%
rfe.ranking_
rfe.support_

#%%
print(X.shape)
X_new = rfe.transform(X)
print(X_new.shape)

#%%""" Feature importances """
model = DecisionTreeClassifier(random_state=seed)
model.fit(X, y)
model.feature_importances_

#%%
plt.barh(y=X.columns, width=model.feature_importances_)
plt.show()

#%%""" Coefs """
model = LogisticRegression()
model.fit(X, y)
model.coef_[0]

#%%
alg = LogisticRegression()
scaler = MinMaxScaler()
model = make_pipeline(scaler, alg)
model.fit(X, y)
# model.coef_[0]

#%%
model.steps[-1][1].coef_[0]

#%%
plt.barh(y=X.columns, width=model.steps[-1][1].coef_[0])
plt.show()

#%% Validacion de modelos
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, cross_val_predict

cv_technique = KFold(n_splits=5, shuffle=True, random_state=seed)
model = LogisticRegression(random_state=seed)
# cross_val_predict(model, X, y, cv=cv_technique)
cross_val_score(model, X, y, cv=cv_technique)

#%%
result = cross_val_score(model, X, y, cv=cv_technique)
result.mean()

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('DTC', DecisionTreeClassifier(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier(random_state=seed)))
models.append(('SVM', SVC(random_state=seed)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))

#%%
seed = 45444

results = []
names = []
for name, model in models:
    cv_technique = KFold(n_splits=10, shuffle=True, random_state=seed)
    result = cross_val_score(model, X, y, cv=cv_technique)

    names.append(name)
    results.append(result)

plt.figure(figsize=(10,5))
plt.boxplot(results)
plt.xticks(range(1, len(names)+1), names)
plt.show()

#%% METRICAS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
model = RandomForestClassifier(random_state=seed)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
confusion_matrix(y_true=y_test,y_pred=y_pred)

#%% Accuracy
accuracy_score(y_true=y_test,y_pred=y_pred)

#%%Tuneado de modelos
from sklearn.model_selection import GridSearchCV

model=DecisionTreeClassifier()
cv_technique= KFold(n_splits=10)

parametros= {
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[3,4,5,6,7,8,9,10],
    'min_samples_split':[2,3,4,5,6,7,8,9,10],
    'random_state':[seed]
}

grid_model = GridSearchCV(estimator=model,param_grid=parametros, cv=cv_technique, scoring='accuracy')

#%% No se mete el train y test debdido a que el gridsearchcv ya hace la division
grid_model.fit(X,y)

#%%
grid_model.best_score_

#%%
grid_model.best_estimator_

#%%
grid_model.predict(X)

#%% Buena praxis es hacer un train test split para controlar overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
grid_model.fit(X_train,y_train)
grid_model.best_score_

#%%
y_pred= grid_model.predict(X_test)
confusion_matrix(y_true=y_test,y_pred=y_pred)

#%%
accuracy_score(y_true=y_test,y_pred=y_pred)

#%%
