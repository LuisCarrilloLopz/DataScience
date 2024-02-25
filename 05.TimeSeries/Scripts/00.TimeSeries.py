"""
Nombre del archivo: 00.TimeSeries.py
Autor: Luis Eduardo Carrillo López
Fecha de creación: 24/02/2024
Fecha de última modificación: 24/02/2024
"""

# Importar librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score


#%% Establecer directorio de trabajo
os.chdir('/Users/luiscarrillo/OneDrive/Desktop/GitHub/DataScience/05.TimeSeries/Datasets/')

#%% Funciones a utilizar
def visualize_adfuller_results(df,series, title, ax,):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen'
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'gold'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'orange'
    else:
        linecolor = 'indianred'
    sns.lineplot(x=df.index, y=series, ax=ax, color=linecolor)
    ax.set_title(f'ADF Statistic {adf_stat:0.3f}, p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic", "p-value",
                                             "Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

######################
#%% Time Series Power Consumption #
######################
powerconsumption = pd.read_csv('powerconsumption.csv', sep=',', parse_dates=[0])

#%% Revisamos el tipo de datos
print('Data types:\n', powerconsumption.dtypes)

#%% Revisamos los primeros registros
print(powerconsumption.head())
print(f'\nRango de fechas: {powerconsumption.Datetime.min()}/{ powerconsumption.Datetime.max()}')

#%% Establecemos la fecha como índice y se elimina
powerconsumption = powerconsumption.set_index('Datetime')
print(powerconsumption.head())

#%% Nos quedamos dos de las variables target y nos centramos en la de "Quads"
powerconsumption = powerconsumption.drop(['PowerConsumption_Zone2','PowerConsumption_Zone3'], axis=1)
targets = ['PowerConsumption_Zone1']
features = [feature for feature in powerconsumption.columns if feature not in targets]
powerconsumption.head()

#%% Graficamos la serie de tiempo
powerconsumption.plot(figsize=(12,6))
plt.show()

#%% Revisar las distribuciones de las variables por medio de histogramas para el mes de diciembre 2017
powerconsumption.hist(figsize=(12,12))
plt.show()

#%% Graficando en detalle cada una de las variables "Features" y Target
f, ax = plt.subplots(nrows=6, ncols=1, figsize=(15, 30))

sns.lineplot(x=powerconsumption.index, y=powerconsumption.Temperature.fillna(np.inf), ax=ax[0], color='dodgerblue')
ax[0].set_title('Feature: Temperature', fontsize=14)
ax[0].set_ylabel(ylabel='Temperature', fontsize=14)

sns.lineplot(x=powerconsumption.index, y=powerconsumption.Humidity.fillna(np.inf), ax=ax[1], color='dodgerblue')
ax[1].set_title('Feature: Humidity', fontsize=14)
ax[1].set_ylabel(ylabel='Humidity', fontsize=14)

sns.lineplot(x=powerconsumption.index, y=powerconsumption.WindSpeed.fillna(np.inf), ax=ax[2], color='dodgerblue')
ax[2].set_title('Feature: WindSpeed', fontsize=14)
ax[2].set_ylabel(ylabel='WindSpeed', fontsize=14)

sns.lineplot(x=powerconsumption.index, y=powerconsumption.GeneralDiffuseFlows.fillna(np.inf), ax=ax[3], color='dodgerblue')
ax[3].set_title('Feature: GeneralDiffuseFlows', fontsize=14)
ax[3].set_ylabel(ylabel='GeneralDiffuseFlows', fontsize=14)

sns.lineplot(x=powerconsumption.index, y=powerconsumption.DiffuseFlows.fillna(np.inf), ax=ax[4], color='dodgerblue')
ax[4].set_title('Feature: DiffuseFlows', fontsize=14)
ax[4].set_ylabel(ylabel='DiffuseFlows', fontsize=14)

sns.lineplot(x=powerconsumption.index, y=powerconsumption.PowerConsumption_Zone1.fillna(np.inf), ax=ax[5], color='dodgerblue')
ax[5].set_title('Target: PowerConsumption Zone1', fontsize=14)
ax[5].set_ylabel(ylabel='PowerConsumption Zone1', fontsize=14)

for i in range(6):
    ax[i] = ax[i].set_xlabel(xlabel='Datetime', fontsize=14)
plt.show()

#%% Nos vamos a quedar con la última semana de diciembre 2017 de los datos para visualizar mejor la proyección.
powerconsumption = powerconsumption['2017-12-24 00:00:00':]
powerconsumption.plot(figsize=(12,6))

t = np.linspace(0, 19, 20)

fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20,4))
stationary = [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6]
sns.lineplot(x=t, y=stationary, ax=ax[0], color='forestgreen')
sns.lineplot(x=t, y=5, ax=ax[0], color='grey')
sns.lineplot(x=t, y=6, ax=ax[0], color='grey')
sns.lineplot(x=t, y=4, ax=ax[0], color='grey')
ax[0].lines[2].set_linestyle("--")
ax[0].lines[3].set_linestyle("--")
ax[0].set_title(f'Estacionario \nmedia constante \nvarianza constante \ncovarianza constante', fontsize=14)

nonstationary1 = [9, 0, 1, 10, 8, 1, 2, 9, 7, 2, 3, 8, 6, 3, 4, 7, 5, 4, 5, 6]
sns.lineplot(x=t, y=nonstationary1, ax=ax[1], color='indianred')
sns.lineplot(x=t, y=5, ax=ax[1], color='grey')
sns.lineplot(x=t, y=t*0.25-0.5, ax=ax[1], color='grey')
sns.lineplot(x=t, y=t*(-0.25)+11, ax=ax[1], color='grey')
ax[1].lines[2].set_linestyle("--")
ax[1].lines[3].set_linestyle("--")
ax[1].set_title(f'No Estacionario \nmedia constante \nvarianza no constante \ncovarianza constante', fontsize=14)

nonstationary2 = [0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11]
sns.lineplot(x=t, y=nonstationary2, ax=ax[2], color='indianred')
sns.lineplot(x=t, y=t*0.5+0.7, ax=ax[2], color='grey')
sns.lineplot(x=t, y=t*0.5, ax=ax[2], color='grey')
sns.lineplot(x=t, y=t*0.5+1.5, ax=ax[2], color='grey')
ax[2].lines[2].set_linestyle("--")
ax[2].lines[3].set_linestyle("--")
ax[2].set_title(f'No Estacionario \nmedia no constante \nvarianza constante \ncovarianza constante', fontsize=14)

nonstationary3 = [5, 4.5, 4, 4.5, 5, 5.5, 6, 5.5, 5, 4.5, 4, 5, 6, 5, 4, 6, 4, 6, 4, 6]
sns.lineplot(x=t, y=nonstationary3, ax=ax[3], color='indianred')
sns.lineplot(x=t, y=5, ax=ax[3], color='grey')
sns.lineplot(x=t, y=6, ax=ax[3], color='grey')
sns.lineplot(x=t, y=4, ax=ax[3], color='grey')
ax[3].lines[2].set_linestyle("--")
ax[3].lines[3].set_linestyle("--")
ax[3].set_title(f'Estacionario \nmedia constante \nvarianza constante \ncovarianza no constante', fontsize=14)

for i in range(4):
    ax[i].set_ylim([-1, 12])
    ax[i].set_xlabel('Tiempo', fontsize=14)

#%% Realizar el test de Adfuller para verificar si la serie es estacionaria
#%% Análsis de autocorrelación
fas = sm.tsa.acf(powerconsumption.PowerConsumption_Zone1, nlags=200)
fap = sm.tsa.pacf(powerconsumption.PowerConsumption_Zone1, nlags=150)

fig, axs = plt.subplots(1, 2, figsize=(15,7))
fig.suptitle('Time series correlograms', y=1)
axs[0].stem(fas)
axs[0].set_title('ACF')
axs[0].set_xlabel('n_lags')
axs[0].grid(True)
axs[1].stem(fap)
axs[1].set_title('PACF')
axs[1].set_xlabel('n_lags')
axs[1].grid(True)
plt.tight_layout()
plt.show()
#%% Movin average
MMd = powerconsumption.PowerConsumption_Zone1.rolling(144, center=True).mean()
STDd = powerconsumption.PowerConsumption_Zone1.rolling(144, center=True).std()

plt.figure(figsize=(18,9))
plt.plot(powerconsumption.PowerConsumption_Zone1, linestyle='-', color='blue', alpha=0.4, label='data')
plt.plot(MMd, linestyle='-', color='red', label='Moving average (1d)')
plt.plot(STDd, linestyle='--', color='black', label='STD (1d)')
plt.ylabel('Power Consumption')
plt.title('Moving average', y=1)
plt.legend()
plt.show()

#%% Dicky Fuller Test
result = adfuller(powerconsumption.PowerConsumption_Zone1)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print(f'Critical Values: {result[4]}')

#%% Visualizar los resultados del test de Adfuller
result = adfuller(powerconsumption.PowerConsumption_Zone1.values)
adf_stat = result[0]
p_val = result[1]

crit_val_1 = result[4]['1%']
crit_val_5 = result[4]['5%']
crit_val_10 = result[4]['10%']

f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
visualize_adfuller_results(powerconsumption,powerconsumption.Temperature.values, 'Temperature', ax[0, 0],)
visualize_adfuller_results(powerconsumption,powerconsumption.Humidity.values, 'Humidity', ax[1, 0])
visualize_adfuller_results(powerconsumption,powerconsumption.WindSpeed.values, 'WindSpeed', ax[0, 1])
visualize_adfuller_results(powerconsumption,powerconsumption.GeneralDiffuseFlows.values, 'GeneralDiffuseFlows', ax[1, 1])
visualize_adfuller_results(powerconsumption,powerconsumption.DiffuseFlows.values, 'DiffuseFlows', ax[2, 0])
visualize_adfuller_results(powerconsumption,powerconsumption.PowerConsumption_Zone1.values, 'PowerConsumption_Zone1', ax[2, 1])

plt.tight_layout()
plt.show()

adf_test(powerconsumption.PowerConsumption_Zone1)

#%% Descomposición de la serie de tiempo
decomposition = seasonal_decompose(powerconsumption.PowerConsumption_Zone1, model='additive', period=144)
fig = decomposition.plot()
plt.xticks(rotation=45)
plt.show()

#%% Obtener dos muestras de powerconsumption la primera de con el 80% de los datos y la segunda con el 20% restante ordenados por fecha
powerconsumption = powerconsumption.sort_index()
train = powerconsumption.iloc[:-202]
train = train[['PowerConsumption_Zone1']]
train.index = pd.DatetimeIndex(train.index.values,
                               freq=train.index.inferred_freq)
test = powerconsumption.tail(202)
test = test[['PowerConsumption_Zone1']]
test.index = pd.DatetimeIndex(test.index.values,
                               freq=test.index.inferred_freq)

#%% Encontrar el modelo de suavizado exponencial mas adecuado para la serie de tiempo con el suavizado Holt-Winters
modelo_holt_winters = sm.tsa.ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=144).fit()
print(modelo_holt_winters.summary())

#%% Obtener predicciones para 1 año
predicciones_hw = modelo_holt_winters.forecast(steps=202)
# Mostrar la descripción del modelo
modelo_holt_winters.summary()

#%%# Crear un gráfico con matplotlib
plt.figure(figsize=(10, 6))
plt.plot(powerconsumption.index, powerconsumption['PowerConsumption_Zone1'], label='Observados',linestyle='-', color='blue')
plt.plot(train.index, modelo_holt_winters.fittedvalues, label='Suavizado Holt-Winters', linestyle='--',
         color='orange')
plt.plot(predicciones_hw.index, predicciones_hw, label='Holt-Winters Prediction', linestyle='--',color='green')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title('Valores Observados, Suavizado Holt-Winters y Predicción')
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.show()

#%%Calcular el accuracy en la muestra de train y test
train['PowerConsumption_Zone1'] = train['PowerConsumption_Zone1'].values
train['fitted'] = modelo_holt_winters.fittedvalues
test['PowerConsumption_Zone1'] = test['PowerConsumption_Zone1'].values
test['forecast'] = predicciones_hw

train['error'] = train['PowerConsumption_Zone1'] - train['fitted']
train['error_pct'] = train['error'] / train['PowerConsumption_Zone1']
test['error'] = test['PowerConsumption_Zone1'] - test['forecast']
test['error_pct'] = test['error'] / test['PowerConsumption_Zone1']

train_accuracy = 1 - np.mean(np.abs(train['error']) / np.abs(train['PowerConsumption_Zone1']))
test_accuracy = 1 - np.mean(np.abs(test['error']) / np.abs(test['PowerConsumption_Zone1']))

print(f'Accuracy en la muestra de train: {train_accuracy:0.2%}')
print(f'Accuracy en la muestra de test: {test_accuracy:0.2%}')

#%% Calcular el RMSE, MSE, MAE, y matriz de confusión
rmse = np.sqrt(mean_squared_error(test['PowerConsumption_Zone1'], test['forecast']))
mse = mean_squared_error(test['PowerConsumption_Zone1'], test['forecast'])
mae = mean_absolute_error(test['PowerConsumption_Zone1'], test['forecast'])

print(f'RMSE: {rmse:0.2f}')
print(f'MSE: {mse:0.2f}')
print(f'MAE: {mae:0.2f}')

#%% Representar la serie y las funciones de autocorrelación y autocorrelación parcial.
fig, ax = plt.subplots(3, 1, figsize=(15, 15))
sm.graphics.tsa.plot_acf(train['error'].iloc[1:], lags=144, ax=ax[0])
sm.graphics.tsa.plot_pacf(train['error'].iloc[1:], lags=144, ax=ax[1])
ax[2].plot(train['error'])
plt.show()

#%% Modelo ARIMA(2,2,1)(0,1,1)144
modelo_arima_221 = sm.tsa.ARIMA(train['PowerConsumption_Zone1'], order=(2, 2, 1), seasonal_order=(0,1,1,144)).fit()
print(modelo_arima_221.summary())

#%% Predicciones
predicciones_arima_221 = modelo_arima_221.forecast(steps=202)

#%% Gráfico
plt.figure(figsize=(10, 6))
plt.plot(powerconsumption.index, powerconsumption['PowerConsumption_Zone1'], label='Observados', linestyle='-', color='blue')
plt.plot(train.index, modelo_arima_221.fittedvalues, label='ARIMA(2,2,1)(0,1,1)144', linestyle='--', color='orange')
plt.plot(predicciones_arima_221, label='ARIMA(2,2,1)(0,1,1)144 Prediction', linestyle='--', color='green')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title('Valores Observados, ARIMA(2,2,1)(0,1,1)144 y Predicción')
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.show()

#%% Calcular el accuracy en la muestra de train y test
train['fitted_arima_221'] = modelo_arima_221.fittedvalues
test['forecast_arima_221'] = predicciones_arima_221

train['error_arima_221'] = train['PowerConsumption_Zone1'] - train['fitted_arima_221']
train['error_pct_arima_221'] = train['error_arima_221'] / train['PowerConsumption_Zone1']
test['error_arima_221'] = test['PowerConsumption_Zone1'] - test['forecast_arima_221']
test['error_pct_arima_221'] = test['error_arima_221'] / test['PowerConsumption_Zone1']

train_accuracy_arima_221 = 1 - np.mean(np.abs(train['error_arima_221']) / np.abs(train['PowerConsumption_Zone1']))
test_accuracy_arima_221 = 1 - np.mean(np.abs(test['error_arima_221']) / np.abs(test['PowerConsumption_Zone1']))

print(f'Accuracy en la muestra de train: {train_accuracy_arima_221:0.2%}')
print(f'Accuracy en la muestra de test: {test_accuracy_arima_221:0.2%}')

#%% Calcular el RMSE, MSE, MAE, y matriz de confusión
rmse_arima_221 = np.sqrt(mean_squared_error(test['PowerConsumption_Zone1'], test['forecast_arima_221']))
mse_arima_221 = mean_squared_error(test['PowerConsumption_Zone1'], test['forecast_arima_221'])
mae_arima_221 = mean_absolute_error(test['PowerConsumption_Zone1'], test['forecast_arima_221'])

print(f'RMSE: {rmse_arima_221:0.2f}')
print(f'MSE: {mse_arima_221:0.2f}')
print(f'MAE: {mae_arima_221:0.2f}')



