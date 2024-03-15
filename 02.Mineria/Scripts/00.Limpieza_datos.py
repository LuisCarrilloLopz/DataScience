# Cargo las librerias que voy a utilizar
import os
import pandas as pd
import numpy as np

from Datasets.FuncionesMineria import analizar_variables_categoricas, atipicosAmissing, patron_perdidos, ImputacionCuant, \
    ImputacionCuali

# %% Establecemos nuestro escritorio de trabajo

os.chdir('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/')

# %% Cargo los datos

df = pd.read_excel('DatosElecciones.xlsx')

with open('FuncionesMineria.py') as file:
    exec(file.read())

# %%  Comprobamos el tipo de formato de las variables variable que se ha asignado en la lectura.
# No todas las categoricas estan como queremos

df.dtypes

# %% Genera una lista con los nombres de las variables.

variables = list(df.columns)

# %% Indico las categóricas que aparecen como numéricas
numericasAcategoricas = ['CodigoProvincia', 'AbstencionAlta', 'Izquierda', 'Derecha']

# %% Las transformo en categóricas

for var in numericasAcategoricas:
    df[var] = df[var].astype(str)

# %% Comprobamos que todas las variables tienen el formato que queremos

df.dtypes

# %% Seleccionar las columnas numéricas del DataFrame

numericas = df.select_dtypes(include=['int', 'int32', 'int64', 'float', 'float32', 'float64']).columns

# %% Seleccionar las columnas categóricas del DataFrame

categoricas = [variable for variable in variables if variable not in numericas]

# %% Frecuencias de los valores en las variables categóricas

a = analizar_variables_categoricas(df)

# %% Descriptivos variables numéricas mediante función describe() de Python

descriptivos_num = df.describe().T

# %% Añadimos más descriptivos a los anteriores

for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = df[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = df[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(df[num].dropna().values)

# %% Muestra valores perdidos

b = df[variables].isna().sum()

# %% Comenzamos la depuración de los datos
# A veces los 'nan' vienen como una cadena de caracteres, los modificamos a perdidos.

for x in categoricas:
    df[x] = df[x].replace('nan', np.nan)

# Missings no declarados variables cualitativas (NSNC, ?)
df['Densidad'] = df['Densidad'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
df['Explotaciones'] = df['Explotaciones'].replace(99999, np.nan)

# Valores fuera de rango
df['Age_19_65_pct'] = [x if 0 <= x <= 100 else np.nan for x in df['Age_19_65_pct']]
df['SameComAutonPtge'] = [x if 0 <= x <= 100 else np.nan for x in df['SameComAutonPtge']]
df['PobChange_pct'] = [x if 0 <= x <= 100 else np.nan for x in df['PobChange_pct']]

# %%  Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
df = df.set_index(['Name', 'CodigoProvincia', 'CCAA'])
varObjCont = df[['AbstentionPtge', 'Izda_Pct', 'Dcha_Pct', 'Otros_Pct']]
varObjBin = df[['AbstencionAlta', 'Izquierda', 'Derecha']]
df_input = df.drop(['AbstentionPtge', 'Izda_Pct', 'Dcha_Pct', 'Otros_Pct', 'AbstencionAlta', 'Izquierda', 'Derecha'],
                   axis=1)

# %% Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(df_input.columns)

# %% Selecionamos las variables numéricas
numericas_input = df_input.select_dtypes(include=['int', 'int32', 'int64', 'float', 'float32', 'float64']).columns

# %% Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]

# %%  ATIPICOS

# Cuento el porcentaje de atipicos de cada variable.

# Seleccionar las columnas numéricas en el DataFrame
# Calcular la proporción de valores atípicos para cada columna numérica
# utilizando una función llamada 'atipicosAmissing'
# 'x' representa el nombre de cada columna numérica mientras se itera a través de 'numericas'
# 'atipicosAmissing(datos_input[x])' es una llamada a una función que devuelve una dupla
# donde el segundo elemento ([1]) es el númeron de valores atípicos
# 'len(datos_input)' es el número total de filas en el DataFrame de entrada
# La proporción de valores atípicos se calcula dividiendo la cantidad de valores atípicos por el número total de filas
resultados = {x: atipicosAmissing(df_input[x])[1] / len(df_input) for x in numericas_input}

# %% Modifico los atipicos como missings
for x in numericas_input:
    df_input[x] = atipicosAmissing(df_input[x])[0]

# %% MISSINGS
# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
patron_perdidos(df_input)

# %%Muestra total de valores perdidos por cada variable
df_input[variables_input].isna().sum()

# %% Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = df_input.isna().sum() / len(df_input)

# %% Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
df_input['prop_missings'] = df_input.isna().mean(axis=1)

# Realizamos un estudio descriptivo básico a la nueva variable
df_input['prop_missings'].describe()

# %% Calculamos el número de valores distintos que tiene la nueva variable
len(df_input['prop_missings'].unique())

# Elimino las observaciones con mas de la mitad de datos missings (no hay ninguna)
eliminar = df_input['prop_missings'] > 0.5
df_input = df_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# %% Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
df_input["prop_missings"] = df_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')

# %% Elimino las variables con más de la mitad de datos missings ( ['PobChange_pct'])
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
df_input = df_input.drop(eliminar, axis=1)

# %% IMPUTACIONES
# Imputo todas las cuantitativas, seleccionar el tipo de imputacion: media, mediana o aleatorio
variables_input = list(df_input.columns)
numericas_input = df_input.select_dtypes(include=['int', 'int32', 'int64', 'float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]

for x in numericas_input:
    df_input[x] = ImputacionCuant(df_input[x], 'aleatorio')

# %% Imputo todas las cualitativas, seleccionar el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    df_input[x] = ImputacionCuali(df_input[x], 'aleatorio')

# Reviso que no queden datos missings
df_input.isna().sum()

# %% Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
datosElecciones = pd.concat([varObjBin, varObjCont, df_input], axis=1)
datosElecciones.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/datosElecciones.pkl')

# %%
