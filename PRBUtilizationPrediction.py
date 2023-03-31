# -*- coding: utf-8 -*-
"""
@author: hadis.ab
"""

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#read stats


data_add=r"D:\Reports_Projects\DS_ML_Tavanafarin_Training\proj15\Input.csv"
data=pd.read_csv(data_add, thousands=',', header=1)


#%%preprocessing
data.describe()
data.isna().sum()
data=data.dropna(subset='4G_PRB_Util_Rate_PDSCH_Avg_IR(#)')
data.isna().sum()
data['Time']=pd.to_datetime(data['Time'])
data_train=data[data['Time']<'2022-09-01 00:00:00']

df=data_train[['Time','4G_PRB_Util_Rate_PDSCH_Avg_IR(#)' ]][data['SITE']=='T6828X'] #work on one sample
df_test=data[(data['Time']>='2022-09-01 00:00:00') & (data['SITE']=='T6828X')][['Time','4G_PRB_Util_Rate_PDSCH_Avg_IR(#)' ]]


df=df.set_index(['Time'])

#check stationary
rolmean=df.rolling('28D').mean()
rolstd=df.rolling('28D').std()
#df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'][1:5].mean()
plt.plot(df, color='blue',label='original')#['Time'],df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
plt.plot(rolmean, color='red',label='rolmean')
plt.plot(rolstd, color='black',label='rolstd')
plt.legend(loc='best')
plt.show(block=False)
#or
dftest=adfuller(df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
print('p-value',dftest[1])

df_Log=np.log(df)
rolmean_Log=df_Log.rolling('28D').mean()
rolstd_LOG=df_Log.rolling('28D').std()
df_stationary=df_Log-rolmean_Log
#df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'][1:5].mean()
plt.plot(df_Log, color='blue',label='original')#['Time'],df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
plt.plot(rolmean_Log, color='red',label='rolmean')
plt.plot(rolstd_LOG, color='black',label='rolstd')
plt.legend(loc='best')
plt.show(block=False)
dftest=adfuller(df_stationary['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
print('p-value',dftest[1])

##Arima
df_stationary_shift=df_stationary-df_stationary.shift()
df_stationary_shift=df_stationary_shift.dropna()
plt.plot(df_stationary_shift)
plt.show()
dftest=adfuller(df_stationary_shift['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
print('p-value',dftest[1])

#ACF / PACF
lag_acf=acf(df_stationary_shift, nlags=20) #q=1
lag_pacf=pacf(df_stationary_shift, nlags=20, method='ols') #p=1
plt.plot(lag_acf)
plt.show()
plt.plot(lag_pacf)
plt.show()


# import warnings
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
#                         FutureWarning)
# warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
#                         FutureWarning)
#
# warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)
df_stationary_train=df_stationary[0:len(df_stationary)-5]
df_stationary_test=df_stationary[len(df_stationary)-5:]

import statsmodels.api as sm
model=sm.tsa.arima.ARIMA(df_stationary_train, order=(1,1,1)) #p d q
results_ARIMA=model.fit()
plt.figure()
plt.plot(df_stationary_train.shift(), color='green')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.show()

predictions_ARIMA=pd.DataFrame(results_ARIMA.fittedvalues, copy=True)
df_pred=pd.DataFrame(results_ARIMA.forecast(steps=len(df_stationary_test)))
df_pred=df_pred.rename(columns={'predicted_mean':'4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'})
predictions_ARIMA=predictions_ARIMA.rename(columns={0:'4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'})
df_pred_total = predictions_ARIMA.append(df_pred)
plt.figure()
plt.plot(df_stationary , label='Original', color='blue')
plt.plot(predictions_ARIMA, color='red', label='ARIMA_Result')
plt.plot(df_pred, color='green',label='Predicted_values')
plt.legend()
plt.show()

#rollback
df_pred_total_RollBack=df_pred_total
df_pred_total_RollBack['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']=df_pred_total_RollBack['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']+rolmean_Log['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']
df_pred_total_RollBack['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']=np.exp(df_pred_total_RollBack['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
plt.figure()
plt.plot(df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
plt.plot(df_pred_total_RollBack['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
plt.show()

df_error=pd.merge(df,df_pred_total_RollBack,how='left', left_on=df.index, right_on=df_pred_total_RollBack.index, suffixes=('_actual','_Predicted'))
MSE=((df_error['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)_actual']-df_error['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)_Predicted'])**2).sum()/len(df_pred_total_RollBack)
df_error.to_csv(r'D:\Reports_Projects\DS_ML_Tavanafarin_Training\proj15\Prediction.csv', index=False)



#
# #rollback
# predictions_ARIMA=pd.DataFrame(results_ARIMA.fittedvalues, copy=True)
# predictions_ARIMA=predictions_ARIMA.rename(columns={0:'4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'})
# predictions_ARIMA['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']=predictions_ARIMA['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']+rolmean_Log['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']
# predictions_ARIMA['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']=np.exp(predictions_ARIMA['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
# plt.figure()
# plt.plot(df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
# plt.plot(predictions_ARIMA['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
# plt.show()
#
#
#
# #-----------------
# predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues, copy=True)
# predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
#
# prediction_ARIMA_log=pd.Series(df_stationary['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'].iloc[0], index=df_stationary.index)
# prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
# prediction_ARIMA=np.exp(prediction_ARIMA_log)
# plt.plot(df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)'])
# plt.plot(prediction_ARIMA)
# results_ARIMA.plot_predict(1,len(df['4G_PRB_Util_Rate_PDSCH_Avg_IR(#)']))
# x=results_ARIMA.forecasts(steps=12)
