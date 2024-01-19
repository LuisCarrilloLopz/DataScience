import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

#%%
os.chdir('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/TimeSeries/Datasets/')

################
## Time Series Bitcoin ##
################
Bitcoin_A = pd.read_excel('bitcoin_A.xlsx')

#%%
Bitcoin_A['DIA'] = pd.to_datetime(Bitcoin_A['DIA'], format='%Y-%m-%d')
print(Bitcoin_A.head())

#%%
print(f'\nRango de fechas: {Bitcoin_A.DIA.min()}/{ Bitcoin_A.DIA.max()}')

#%%# se establece la columna fecha como índice y se elimina
Bitcoin_A.index = Bitcoin_A['DIA']
del Bitcoin_A['DIA']
print(Bitcoin_A.head())
sns.lineplot(Bitcoin_A)
plt.xticks(rotation=30, ha='right')

#%%
plt.plot(Bitcoin_A["N_TRANS"])
plt.xticks(rotation=30, ha='right')

#%%
sns.lineplot(Bitcoin_A["PRECIO"])
plt.xticks(rotation=30, ha='right')

#%% 1. Crear un gráfico con dos ejes y compartir el eje x
fig, ax1 = plt.subplots()

# 2. Primera serie en el eje izquierdo (ax1)
color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('PRECIO', color=color)
ax1.plot(Bitcoin_A['PRECIO'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 3. Segunda serie en el eje derecho (ax2)
ax2 = ax1.twinx() # Compartir el eje x
color = 'tab:blue'
ax2.set_ylabel('N_TRANS', color=color)
ax2.plot(Bitcoin_A['N_TRANS'],color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 4. Ajustes de diseño
plt.title('Two Time Series with Different Scales')
plt.show()

#%%
precio_R = Bitcoin_A.PRECIO.loc['2019-08-01':]
precio_R.index.freq = 'D'
# Aplicar suavizado exponencial simple.
modelo_ses = sm.tsa.SimpleExpSmoothing(precio_R , initialization_method="estimated").fit()
# Para seleccionar distintas alphas, fit(smoothing_level=alpha)
# Calcular la predicción para 7 días.
precio_s1 = modelo_ses.forecast(steps=7)
# Para ver parámetros, e.g.: alpha = 0.995
modelo_ses.summary()

#%% Crear un gráfico con matplotlib
plt.figure(figsize=(10, 6))
# Valores observados
plt.plot(precio_R.index, precio_R, label='Observados', marker='x', linestyle='-', color='blue')
# Valores suavizados (fitted)
plt.plot(precio_R.index, modelo_ses.fittedvalues, label='Suavizado', linestyle='--', color='orange')
# Predicción para 7 días
plt.plot(precio_s1.index, precio_s1, label='Predicción para 7 días', linestyle='--', marker='o',
         color='green')
plt.xlabel('Día')
plt.ylabel('PRECIO')
plt.title('Valores Observados, Suavizado y Predicción para 7 días')
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.show()

#%% Aplicar suavizado exponencial doble (Holt).
modelo_holt = sm.tsa.ExponentialSmoothing(precio_R, trend='add', damped=False).fit()
# Obtener predicciones para 7 días
predicciones = modelo_holt.forecast(steps=7)
# Mostrar la descripción del modelo
modelo_holt.summary()

#%% Crear un gráfico con matplotlib
plt.figure(figsize=(10, 6))
plt.plot(precio_R.index, precio_R, label='Observados',
         marker='x', linestyle='-', color='blue')
plt.plot(precio_R.index, modelo_holt.fittedvalues,
         label='Suavizado Doble', linestyle='--', color='orange')
plt.plot(predicciones.index, predicciones, label='Holt',
         linestyle='--',color='green')
plt.plot(precio_s1.index, precio_s1, label='Simple',
         linestyle='--', color='red')
plt.xlabel('Día')
plt.ylabel('PRECIO')
plt.title('Valores Observados, Suavizado Doble y Predicción para 7 días')
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.show()

#%%
################
## Time Series Cordoba ##
################

v_cordoba = pd.read_excel("Cordoba.xlsx")
v_cordoba.head()

#%%
v_cordoba.columns=['Date','V_Resident','V_Extranj']
v_cordoba['Date'] = pd.to_datetime(v_cordoba['Date'], format='%YM%m')
v_cordoba.index = v_cordoba['Date']
del v_cordoba['Date']
print(v_cordoba.head())

#%%
sns.lineplot(v_cordoba)
plt.xticks(rotation=30, ha='right')
plt.title('Viajeros alojados en hoteles en Córdoba')

#%%
result = seasonal_decompose(v_cordoba['V_Extranj'], model='multiplicative')
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(result.observed, label='Observado')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Tendencia')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(result.seasonal,label='Estacionalidad')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(result.resid,label='Residual')
plt.legend(loc='upper left')
plt.show()
#%%
result.plot()

#%% porporcion de reducción del desajuste del modelo creado vs el modelo original
print(result.seasonal)

#%%
plt.plot(result.observed/result.seasonal, label='Ajustada Estacionalidad', color='red')
plt.plot(result.trend, label='Tendencia', color='blue')
plt.plot(result.observed, label='Observado', color='black')
plt.legend()
plt.title("Descomposición de la serie de tiempo")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de viajeros")
plt.show()

#%%
v_cordoba['Año'] = pd.to_datetime(v_cordoba.index, format='%YM%m').year
plt.figure(figsize=(12, 8))
for Año, datos_año in v_cordoba.groupby('Año'):
    plt.plot(datos_año.index.month, datos_año['V_Extranj'], label=str(Año))
# Añadir leyendas y título
plt.legend(title='Año')
plt.title('Seasonal Plot Viajeros Extranjeros en Córdoba')
plt.xlabel('Mes')
plt.ylabel('Viajeros Extranjeros')
# Mostrar el gráfico
plt.show()

#%% En el ejemplo de viajeros de córdoba
# Aplicar suavizado Holt-Winters.
modelo_holt_winters = sm.tsa.ExponentialSmoothing(v_cordoba['V_Extranj'], trend='add',
                                                  seasonal='multiplicative', seasonal_periods=12).fit()
# Obtener predicciones para 1 año
predicciones_hw = modelo_holt_winters.forecast(steps=12)
# Mostrar la descripción del modelo
modelo_holt_winters.summary()

#%%En el ejemplo de viajeros de córdoba
# Crear un gráfico con matplotlib
plt.figure(figsize=(10, 6))
plt.plot(v_cordoba.index, v_cordoba['V_Extranj'], label='Observados', marker='x', linestyle='-', color='blue')
plt.plot(v_cordoba.index, modelo_holt_winters.fittedvalues, label='Suavizado Holt-Winters', linestyle='--',
         color='orange')
plt.plot(predicciones_hw.index, predicciones_hw, label='Holt-Winters', linestyle='--',color='green')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title('Valores Observados, Suavizado Holt-Winters y Predicción')
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.show()

#%%