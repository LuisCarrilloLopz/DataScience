# Cargo las librerias que voy a utilizar
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

#%% Establecemos nuestro escritorio de trabajo

os.chdir('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/')

#%% Cargo los datos
df = pd.read_pickle('DatosElecciones.pkl')
with open('FuncionesMineria.py') as file:
    exec(file.read())

#%%Defino las variables objetivo y las elimino del conjunto de datos input
varObjCont1 = df['AbstentionPtge'] #Continua
varObjCont2 = df['Izda_Pct'] #Continua
varObjCont3 = df['Dcha_Pct'] #Continua
varObjCont4 = df['Otros_Pct'] #Continua

varObjBin1 = df['AbstencionAlta'] #Dicotómica
varObjBin2 = df['Izquierda'] #Dicotómica
varObjBin3 = df['Derecha'] #Dicotómica

df_input = df.drop(['AbstentionPtge','Izda_Pct','Dcha_Pct','Otros_Pct','AbstencionAlta','Izquierda','Derecha'], axis = 1)

#%% Genera una lista con los nombres de las variables.
variables = list(df_input.columns)

#%% Obtengo la importancia de las variables
objetivos_bin = [varObjBin1, varObjBin2, varObjBin3]
objetivos_cont = [varObjCont1, varObjCont2, varObjCont3, varObjCont4]

for objetivo in objetivos_bin + objetivos_cont:
    graficoVcramer(df_input, objetivo)

#%% Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables:
    for objetivo in objetivos_cont:
        v_cramer = Vcramer(df_input[variable], objetivo)
        VCramer = VCramer.append({'Variable': variable, 'Objetivo': objetivo.name, 'Vcramer': v_cramer},
                                 ignore_index=True)

    for objetivo in objetivos_bin:
        v_cramer = Vcramer(df_input[variable], objetivo)
        VCramer = VCramer.append({'Variable': variable, 'Objetivo': objetivo.name, 'Vcramer': v_cramer},
                                 ignore_index=True)

VCramer['Tipo_Variable'] = VCramer['Variable'].apply(lambda x: df_input[x].dtype)
VCramer['Tipo_Objetivo'] = VCramer['Objetivo'].apply(lambda x: df[x].dtype)

#%% Agrupar por 'Objetivo' y aplicar nlargest en 'Vcramer'
resultado = VCramer.groupby('Objetivo')['Vcramer'].nlargest(2)

# Restablecer el índice
resultado = resultado.reset_index()

# Obtener los nombres de las variables correspondientes
resultado['Variable'] = resultado.apply(lambda row: VCramer.loc[row['level_1'], 'Variable'], axis=1)

# Eliminar la columna 'level_1'
resultado = resultado.drop(columns='level_1')

# %%Veo graficamente el efecto de dos variables cuantitativas sobre la binaria1
boxplot_targetbinaria(df_input['totalEmpresas'], varObjBin1, nombre_ejeX='AbstencionAlta', nombre_ejeY='totalEmpresas')
boxplot_targetbinaria(df_input['Pob2010'], varObjBin1, nombre_ejeX='AbstencionAlta', nombre_ejeY='Pob2010')

hist_targetbinaria(df_input['totalEmpresas'], varObjBin1, 'totalEmpresas')
hist_targetbinaria(df_input['Pob2010'], varObjBin1, 'Pob2010')

# %%Veo graficamente el efecto de dos variables cuantitativas sobre la binaria2
boxplot_targetbinaria(df_input['UnemployLess25_Ptge'], varObjBin2, nombre_ejeX='Izquierda', nombre_ejeY='UnemployLess25_Ptge')
boxplot_targetbinaria(df_input['AgricultureUnemploymentPtge'], varObjBin2, nombre_ejeX='Izquierda', nombre_ejeY='AgricultureUnemploymentPtge')

hist_targetbinaria(df_input['UnemployLess25_Ptge'], varObjBin2, 'UnemployLess25_Ptge')
hist_targetbinaria(df_input['AgricultureUnemploymentPtge'], varObjBin2, 'AgricultureUnemploymentPtge')

# %%Veo graficamente el efecto de dos variables cuantitativas sobre la binaria3
boxplot_targetbinaria(df_input['Age_over65_pct'], varObjBin3, nombre_ejeX='Derecha', nombre_ejeY='Age_over65_pct')
boxplot_targetbinaria(df_input['Age_under19_Ptge'], varObjBin3, nombre_ejeX='Derecha', nombre_ejeY='Age_under19_Ptge')

hist_targetbinaria(df_input['Age_over65_pct'], varObjBin3, 'Age_over65_pct')
hist_targetbinaria(df_input['Age_under19_Ptge'], varObjBin3, 'Age_under19_Ptge')

#%%# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = df_input.select_dtypes(include=['int', 'float']).columns

# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)

# Para cada variable objetivo, calcular la matriz de correlación y mostrarla
for objetivo in objetivos_cont:
    # Calcular la matriz de correlación de Pearson
    matriz_corr = pd.concat([objetivo, df_input[numericas]], axis = 1).corr(method = 'pearson')

    # Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))

    # Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
    plt.figure(figsize=(8, 6))

    # Crear un mapa de calor (heatmap) de la matriz de correlación
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)

    # Establecer el título del gráfico
    plt.title(f"Matriz de correlación para {objetivo.name}")

    # Mostrar el gráfico de la matriz de correlación
    plt.show()

#%% Busco las mejores transformaciones para las variables numericas con respecto a los dos tipos de variables
input_cont1 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjCont1)], axis = 1)
input_cont2 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjCont2)], axis = 1)
input_cont3 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjCont3)], axis = 1)
input_cont4 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjCont4)], axis = 1)

#%% Busco las mejores transformaciones para las variables numericas con respecto a los dos tipos de variables
#input_bin1 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjBin1)], axis = 1)
#input_bin2 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjBin2)], axis = 1)
#input_bin3 = pd.concat([df_input, Transf_Auto(df_input[numericas], varObjBin3)], axis = 1)

#%%  Creamos conjuntos de datos que contengan las variables explicativas y una de las variables objetivo y los guardamos
todo_cont1 = pd.concat([input_cont1, varObjCont1], axis = 1)
todo_cont2 = pd.concat([input_cont2, varObjCont2], axis = 1)
todo_cont3 = pd.concat([input_cont3, varObjCont3], axis = 1)
todo_cont4 = pd.concat([input_cont4, varObjCont4], axis = 1)

#%%
#todo_bin1 = pd.concat([input_bin1, varObjBin1], axis = 1)
#todo_bin2 = pd.concat([input_bin2, varObjBin2], axis = 1)
#todo_bin3 = pd.concat([input_bin3, varObjBin3], axis = 1)

#%%
todo_cont1.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Cont/todo_cont1.pkl')
todo_cont2.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Cont/todo_cont2.pkl')
todo_cont3.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Cont/todo_cont3.pkl')
todo_cont4.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Cont/todo_cont4.pkl')

#%%
#todo_bin1.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Bin/todo_bin1.pkl')
#todo_bin2.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Bin/todo_bin2.pkl')
#todo_bin3.to_pickle('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/Mineria/Data/Bin/todo_bin3.pkl')

#%% Comenzamos con la regresion lineal
# Obtengo la particion para la variable objetivo continua elegida (Izda_Pct - varObjCont2)
x_train, x_test, y_train, y_test = train_test_split(df_input, np.ravel(varObjCont2), test_size = 0.2, random_state = 123456)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = ['Population', 'TotalCensus', 'Age_0-4_Ptge', 'Age_under19_Ptge',
             'Age_19_65_pct', 'Age_over65_pct', 'WomanPopulationPtge',
             'ForeignersPtge', 'SameComAutonPtge', 'SameComAutonDiffProvPtge',
             'DifComAutonPtge', 'UnemployLess25_Ptge', 'Unemploy25_40_Ptge',
             'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge',
             'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge',
             'ServicesUnemploymentPtge', 'totalEmpresas', 'Industria',
             'Construccion', 'ComercTTEHosteleria', 'Servicios', 'inmuebles',
             'Pob2010', 'SUPERFICIE', 'PersonasInmueble', 'Explotaciones']
var_categ1 = ['ActividadPpal', 'Densidad']

#%% Creo el modelo
modelo1 = lm(y_train, x_train, var_cont1, var_categ1)
# Visualizamos los resultado del modelo
modelo1['Modelo'].summary()

# El coeficiente sqft_above = 86.2606. Por lo que aumentar una unidad de esta variable
# produce un aumento esperado de 86.2606 unidades en el precio de la vivienda

# El coeficiente floors_1.5 = 7422.7473. Por lo que el precio de la vivienda esperado
# se vee aumentado en 7422.7473 unidades en comparación con la categoria
# de floors "1.0".

#%% Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo1['Modelo'], y_train, modelo1['X'])

#%% Preparamos los datos test para usar en el modelo
x_test_modelo1 = crear_data_modelo(x_test, var_cont1, var_categ1)
# Calculamos la medida de ajuste R^2 para los datos test
print(Rsq(modelo1['Modelo'], y_test, x_test_modelo1))
print(len(modelo1['Modelo'].params))

# %%Nos fijamos en la importancia de las variables
# Parece que no hay ninunga imprescindible
modelEffectSizes(modelo1, y_train, x_train, var_cont1, var_categ1)

#%% Vamos a probar un modelo con menos variables. Recuerdo el grafico de Cramer
graficoVcramer(df_input, varObjCont2) # Pruebo con las mas importantes

VCramer2 = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

variables = list(df_input.columns)
for variable in variables:
    v_cramer2 = Vcramer(df_input[variable], varObjCont2)
    VCramer2 = VCramer2.append({'Variable': variable, 'Objetivo': varObjCont2.name, 'Vcramer': v_cramer2},
                             ignore_index=True)

VCramer2['Tipo_Variable'] = VCramer2['Variable'].apply(lambda x: df_input[x].dtype)

#%% Construyo el segundo modelo
var_cont2 = ['UnemployLess25_Ptge', 'AgricultureUnemploymentPtge', 'SameComAutonDiffProvPtge']
var_categ2 = ['ActividadPpal']

#%% Creo el modelo
modelo2 = lm(y_train, x_train, var_cont2, var_categ2)
modelo2['Modelo'].summary()

#%% Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo2['Modelo'], y_train, modelo2['X'])

# %%Nos fijamos en la importancia de las variables
# Parece que no hay ninunga imprescindible
modelEffectSizes(modelo2, y_train, x_train, var_cont2, var_categ2)

#%% Preparamos los datos test para usar en el modelo
x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)

#%%Calculamos la medida de ajuste R^2 para los datos test
print(Rsq(modelo1['Modelo'], y_test, x_test_modelo1))
print(Rsq(modelo2['Modelo'], y_test, x_test_modelo2))
print(len(modelo1['Modelo'].params))
print(len(modelo2['Modelo'].params))

#%% Construyo el tercer modelo
var_cont3 = ['UnemployLess25_Ptge', 'AgricultureUnemploymentPtge', 'SameComAutonDiffProvPtge','SUPERFICIE']
var_categ3 = ['ActividadPpal','Densidad']

#%% Creo el modelo
modelo3 = lm(y_train, x_train, var_cont3, var_categ3)
modelo3['Modelo'].summary()

#%% Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo3['Modelo'], y_train, modelo3['X'])

# %%Nos fijamos en la importancia de las variables
# Parece que no hay ninunga imprescindible
modelEffectSizes(modelo3, y_train, x_train, var_cont3, var_categ3)

#%% Preparamos los datos test para usar en el modelo
x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)

#%%Calculamos la medida de ajuste R^2 para los datos test
print(Rsq(modelo1['Modelo'], y_test, x_test_modelo1))
print(Rsq(modelo2['Modelo'], y_test, x_test_modelo2))
print(Rsq(modelo3['Modelo'], y_test, x_test_modelo3))
print(len(modelo1['Modelo'].params))
print(len(modelo2['Modelo'].params))
print(len(modelo3['Modelo'].params))

#%%# Pruebo con una interaccion sobre el anterior
# Se podrian probar todas las interacciones dos a dos
var_cont4 = ['UnemployLess25_Ptge', 'AgricultureUnemploymentPtge', 'SameComAutonDiffProvPtge','SUPERFICIE']
var_categ4 = ['ActividadPpal','Densidad']
var_interac4 = [('SameComAutonDiffProvPtge', 'ActividadPpal')]
modelo4 = lm(y_train, x_train, var_cont4, var_categ4, var_interac4)
modelo4['Modelo'].summary()
Rsq(modelo4['Modelo'], y_train, modelo4['X'])
x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
Rsq(modelo4['Modelo'], y_test, x_test_modelo4)
#%%

#%% Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})

#%% Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(200):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_lm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_lm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_lm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_lm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)

    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'Rsquared': modelo1VC + modelo2VC + modelo3VC + modelo4VC,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 4,  # Etiqueta de repetición
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5  # Etiqueta de modelo (1, 2, 3 o 4)
    })

    # Concatena los resultados de esta repetición al DataFrame principal 'results'
    results = pd.concat([results, results_rep], axis=0)


#%% Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de R-squared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico

#%% Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
# Contar el número de parámetros en cada modelo
num_parametros = [len(modelo1['Modelo'].params), len(modelo2['Modelo'].params),
                  len(modelo3['Modelo'].params), len(modelo4['Modelo'].params)]

#%% Teniendo en cuenta el R2, la estabilidad y el numero de parametros, nos quedamos con el modelo3
# Vemos los coeficientes del modelo ganador
modelo3['Modelo'].summary()

#%% Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
print(Rsq(modelo3['Modelo'], y_train, modelo3['X']))
print(Rsq(modelo3['Modelo'], y_test, x_test_modelo3))

#%% Vemos las variables mas importantes del modelo ganador
modelEffectSizes(modelo3, y_train, x_train, var_cont3, var_categ3)


#%%
