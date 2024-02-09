# Cargo las librerias que voy a utilizar
import os
import pandas as pd
from optbinning import Scorecard, BinningProcess, OptimalBinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pysal.lib import weights
from pysal.explore import esda

#%% Llama a las funciones que vamos a utilizar
with open('/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/Scoring/Scoring.py') as file:
    exec(file.read())

# %% Establecemos nuestro escritorio de trabajo
os.chdir('/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/Scoring/Data/')

###############
## GERMAN CREDIT ##
###############
# %% Cargamos los datos
df=pd.read_csv('germancredit.csv')

#%%Recodifico esta variable creditability (variable objetivo) para que sea binaria
df["y"]=0
df.loc[df["creditability"]=="good",["y"]]=0
df.loc[df["creditability"]=="bad", ["y"]]=1
df.drop(labels='creditability',inplace=True, axis=1)

#%% Creo la muestra de entrenamiento y de test
df_train, df_test = train_test_split(df, stratify= df["y"], test_size=.25, random_state=1234)

#%% Realizamos la trimificación optima de age.in.years
variable="age.in.years"
X=df_train[variable].values
Y=df_train['y'].values
optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
optb.fit(X, Y)
optb.splits
binning_table = optb.binning_table
binning_table.build()
#%%

###############
#### PRACTICA ####
###############
# %% Cargamos los datos
df=pd.read_excel('DatosPractica_Scoring.xlsx')
variables = list(df.columns)
#%% Variables a revisar
features = ['ID', 'Cardhldr', 'Age', 'Income', 'Exp_Inc', 'Avgexp', 'Ownrent', 'Selfempl', 'Depndt', 'Inc_per', 'Cur_add', 'Major','Active']

#%% Calculamos el IV para cada variable predictora
for feature in features:
    calc_iv(df, feature, 'default')

#%% # Preprocesamiento de los datos
df['default'].fillna(0, inplace=True)  # Asumimos que los clientes a los que se les negó el crédito no hubieran impagado

#%% Dividir los datos en conjuntos de entrenamiento y prueba
train = df[df['Cardhldr'].notna()].copy()
test = df[df['Cardhldr'].isna()]

#%%
predictors = ['Age', 'Income', 'Exp_Inc', 'Avgexp', 'Ownrent', 'Selfempl', 'Depndt', 'Inc_per', 'Cur_add', 'Major', 'Active']
target = 'default'

X_train = train[predictors]
y_train = train[target]
X_test = test[predictors]

#%% Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#%% Hacer predicciones para los nuevos clientes
predictions = model.predict(X_test)

# Imprimir los IDs de los clientes a los que se les concedería el crédito
print("Crédito concedido a los clientes con los siguientes IDs:")
for i in range(len(predictions)):
    if predictions[i] == 0:
        print(test['ID'].iloc[i])

#%%
gdfm =gpd.read_file("/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/Scoring/Data/Munic04_ESP.shp")
gdfm_Madrid =gdfm[gdfm['COD_PROV']=='28']
gdfm_Madrid.explore(column='PrecioIn16',scheme='NaturalBreaks',k=9, cmap='YlOrRd',legend=False,style_kwds=dict(fillOpacity=0.8))
median = gdfm_Madrid.groupby('MUN')['PrecioIn16'].median()
#%%
from shapely.geometry import MultiPolygon
gdfm =gpd.read_file("/Users/luiscarrillo/Library/CloudStorage/OneDrive-Personal/Desktop/GitHub/DataScience/Scoring/Data/Munic04_ESP.shp")
wq = weights.contiguity.Queen.from_dataframe(gdfm)
wq.transform = "R"

#%%
moran = esda.moran.Moran(gdfm["TASA_PARO"], wq)
#%%

1) a, c, d
2) c, e
3) b, e
4) b, c, e
5) b, e
6) e, f, h
7) c
8)
Age Respuesta 1,165215904512587
Income Respuesta 0,4729580246241282
Exp_Inc Respuesta 0,03132782158113419
Avgexp Respuesta 0,827922217075151
Ownrent Respuesta 0,036974320601192254
Selfempl Respuesta 0,005128927253047892
Depndt Respuesta 0,03028479636388836
Inc_per Respuesta 0,6550471756019756
Cur_add Respuesta 0,6192197085012037
Major Respuesta 0,05681305499334241
Active Respuesta 0,4348202577222598
9) TODOS
10) b, c, e, f, g,h
11) c, e, f, g
12) c
13) c, f
14) c, d, e
